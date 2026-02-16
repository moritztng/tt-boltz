"""Boltz-2 structure prediction CLI."""

import json
import os
import random
import tarfile
import time
import traceback
import urllib.request
import warnings
from dataclasses import replace
from functools import partial
from pathlib import Path

import click
import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

from tt_boltz.data import const
from tt_boltz.data.featurizer import Boltz2Featurizer
from tt_boltz.data.mol import load_canonicals, load_molecules
from tt_boltz.data.msa import run_mmseqs2
from tt_boltz.data.parse import parse_a3m, parse_csv, parse_fasta, parse_yaml
from tt_boltz.data.tokenize import Boltz2Tokenizer
from tt_boltz.data.types import Coords, Input, Interface
from tt_boltz.data.write import to_mmcif, to_pdb
from tt_boltz.boltz2 import Boltz2

URLS = {
    "mols": "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar",
    "conf": ["https://model-gateway.boltz.bio/boltz2_conf.ckpt",
             "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt"],
    "aff": ["https://model-gateway.boltz.bio/boltz2_aff.ckpt",
            "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt"],
}


def download(urls: list[str], dest: Path) -> None:
    """Download file from list of URLs (tries each until success)."""
    if dest.exists():
        return
    click.echo(f"Downloading {dest.name}")
    for url in urls:
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception:
            continue
    raise RuntimeError(f"Failed to download {dest.name}")


def download_all(cache: Path) -> None:
    """Download all required model files and molecules."""
    tar_path = cache / "mols.tar"
    if not tar_path.exists():
        urllib.request.urlretrieve(URLS["mols"], tar_path)
    if not (cache / "mols").exists():
        with tarfile.open(tar_path) as tar:
            tar.extractall(cache)
    download(URLS["conf"], cache / "boltz2_conf.ckpt")
    download(URLS["aff"], cache / "boltz2_aff.ckpt")


def compute_msa(seqs: dict[str, str], target_id: str, msa_dir: Path, url: str, strategy: str,
                username: str = None, password: str = None, api_key: str = None) -> None:
    """Generate MSAs for protein sequences."""
    click.echo(f"MSA for {target_id} ({len(seqs)} sequences)")
    headers = {"Content-Type": "application/json", "X-API-Key": api_key} if api_key else None
    seqs_list = list(seqs.values())

    paired = (run_mmseqs2(seqs_list, msa_dir / f"{target_id}_paired_tmp", use_env=True,
                         use_pairing=True, host_url=url, pairing_strategy=strategy,
                         msa_server_username=username, msa_server_password=password, auth_headers=headers)
             if len(seqs) > 1 else [""] * len(seqs))

    unpaired = run_mmseqs2(seqs_list, msa_dir / f"{target_id}_unpaired_tmp", use_env=True,
                          use_pairing=False, host_url=url, pairing_strategy=strategy,
                          msa_server_username=username, msa_server_password=password, auth_headers=headers)

    for i, name in enumerate(seqs):
        paired_seqs = [s for s in paired[i].strip().splitlines()[1::2][:const.max_paired_seqs] if s != "-" * len(s)]
        unpaired_seqs = unpaired[i].strip().splitlines()[1::2][:const.max_msa_seqs - len(paired_seqs)]
        if paired_seqs:
            unpaired_seqs = unpaired_seqs[1:]
        keys = list(range(len(paired_seqs))) + [-1] * len(unpaired_seqs)
        lines = ["key,sequence"] + [f"{k},{s}" for k, s in zip(keys, paired_seqs + unpaired_seqs)]
        (msa_dir / f"{name}.csv").write_text("\n".join(lines))


def prepare_features(path, ccd, mol_dir, msa_dir, tokenizer, featurizer,
                     use_msa, msa_url, msa_strategy, msa_user, msa_pass, api_key,
                     max_msa, method=None, affinity=False, pred_structure=None):
    """Parse, resolve MSA, tokenize, featurize — all in memory.

    Only MSA CSV files touch disk (cached in msa_dir).
    Returns (features_dict, input_structure).
    """
    suffix = path.suffix.lower()
    if suffix in (".fa", ".fas", ".fasta"):
        target = parse_fasta(path, ccd, mol_dir, True)
    elif suffix in (".yml", ".yaml"):
        target = parse_yaml(path, ccd, mol_dir, True)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    record = target.record
    struct = pred_structure if pred_structure is not None else target.structure

    # Identify protein chains needing MSA
    to_gen = {}
    for chain in record.chains:
        if chain.mol_type == const.chain_type_ids["PROTEIN"] and chain.msa_id == 0:
            msa_name = f"{record.id}_{chain.entity_id}"
            to_gen[msa_name] = target.sequences[chain.entity_id]
            chain.msa_id = str(msa_dir / f"{msa_name}.csv")
        elif chain.msa_id == 0:
            chain.msa_id = -1

    # Generate MSA if not cached
    if to_gen and not all((msa_dir / f"{k}.csv").exists() for k in to_gen):
        if not use_msa:
            raise RuntimeError("Missing MSAs, use --use_msa_server")
        compute_msa(to_gen, record.id, msa_dir, msa_url, msa_strategy, msa_user, msa_pass, api_key)

    # Parse MSAs in memory (deduplicated by path)
    msa_cache = {}
    msas = {}
    for chain in record.chains:
        if chain.msa_id == -1:
            continue
        key = str(chain.msa_id)
        if key not in msa_cache:
            p = Path(key)
            msa_cache[key] = parse_a3m(p, None, max_msa) if p.suffix == ".a3m" else parse_csv(p, max_msa)
        msas[chain.chain_id] = msa_cache[key]

    # Build Input and tokenize
    templates = target.templates if target.templates else None
    inp = Input(struct, msas, record=record, residue_constraints=target.residue_constraints,
                templates=templates, extra_mols=target.extra_mols)
    tok = tokenizer.tokenize(inp)

    # Affinity cropping
    if affinity:
        td, tb = tok.tokens, tok.bonds
        valid = td[td["resolved_mask"]]
        lig_coords = valid[valid["affinity_mask"]]["center_coords"]
        dists = np.min(np.sum((valid["center_coords"][:, None] - lig_coords[None])**2, -1)**0.5, axis=1)

        cropped, atoms, prot = set(), 0, set()
        lig_ids = set(valid[valid["mol_type"] == const.chain_type_ids["NONPOLYMER"]]["token_idx"])

        for idx in np.argsort(dists):
            token = valid[idx]
            chain_tokens = td[td["asym_id"] == token["asym_id"]]
            if len(chain_tokens) <= 10:
                neighbors = chain_tokens
            else:
                res_window = chain_tokens[(chain_tokens["res_idx"] >= token["res_idx"] - 10) &
                                         (chain_tokens["res_idx"] <= token["res_idx"] + 10)]
                neighbors = res_window[res_window["res_idx"] == token["res_idx"]]
                mi = ma = token["res_idx"]
                while neighbors.size < 10:
                    mi -= 1; ma += 1
                    neighbors = res_window[(res_window["res_idx"] >= mi) & (res_window["res_idx"] <= ma)]

            new_ids = set(neighbors["token_idx"]) - cropped
            new_atoms = np.sum(td[list(new_ids)]["atom_num"])
            if (len(new_ids) > 256 - len(cropped) or atoms + new_atoms > 2048 or
                len(prot | new_ids - lig_ids) > 200):
                break
            cropped.update(new_ids)
            atoms += new_atoms
            prot.update(new_ids - lig_ids)

        td = td[sorted(cropped)]
        tb = tb[np.isin(tb["token_1"], td["token_idx"]) & np.isin(tb["token_2"], td["token_idx"])]
        tok = replace(tok, tokens=td, bonds=tb)

    # Load molecules
    mols = {**ccd, **target.extra_mols}
    needed = set(tok.tokens["res_name"].tolist()) - set(mols)
    mols.update(load_molecules(mol_dir, needed))

    # Constraints
    opts = record.inference_options
    pocket, contact = (opts.pocket_constraints, opts.contact_constraints) if opts else (None, None)

    # Featurize
    feats = featurizer.process(
        tok, np.random.default_rng(42), mols, False, const.max_msa_seqs,
        pad_to_max_seqs=False, single_sequence_prop=0.0, compute_frames=True,
        inference_pocket_constraints=pocket, inference_contact_constraints=contact,
        compute_constraint_features=True, override_method=method, compute_affinity=affinity
    )
    feats["record"] = record
    return feats, target.structure


def to_batch(feats: dict, device: torch.device) -> dict:
    """Convert features to batch format on device."""
    skip = {"all_coords", "all_resolved_mask", "crop_to_all_atom_map", "chain_symmetries",
            "amino_acids_symmetries", "ligand_symmetries", "record", "affinity_mw"}
    batch = {}
    for k, v in feats.items():
        if k in skip:
            batch[k] = [v] if k == "record" else v
        elif hasattr(v, 'unsqueeze'):
            batch[k] = v.unsqueeze(0).to(device)
        else:
            batch[k] = v
    return batch


def write_result(pred, batch, input_struct, out_dir, fmt,
                 write_pae=False, write_pde=False, write_embeddings=False):
    """Write CIF/PDB structure files. Return (metrics_dict, best_structure).

    pLDDT embedded in B-factors. All confidence values returned in metrics dict.
    """
    if pred["exception"]:
        return None, None

    record = batch["record"][0]
    struct = input_struct.remove_invalid_chains()

    confidence = pred.get("confidence_score", torch.zeros(1))
    rank = {i.item(): r for r, i in enumerate(torch.argsort(confidence, descending=True))}
    mask_1d = pred["masks"].squeeze(0) if pred["masks"].dim() > 1 else pred["masks"]

    best_struct = None
    write_fn = to_pdb if fmt == "pdb" else to_mmcif

    for model_idx in range(pred["coords"].shape[0]):
        model_rank = rank.get(model_idx, model_idx)
        coords = pred["coords"][model_idx][mask_1d.bool()].cpu().numpy()

        atoms, residues = struct.atoms.copy(), struct.residues.copy()
        atoms["coords"], atoms["is_present"] = coords, True
        residues["is_present"] = True
        new_struct = replace(struct, atoms=atoms, residues=residues,
                            interfaces=np.array([], dtype=Interface),
                            coords=np.array([(x,) for x in coords], dtype=Coords))

        plddt = pred.get("plddt", [None] * (model_idx + 1))[model_idx]

        if model_rank == 0:
            best_struct = new_struct
            (out_dir / f"{record.id}.{fmt}").write_text(write_fn(new_struct, plddt, True))
        else:
            (out_dir / f"{record.id}_model_{model_rank}.{fmt}").write_text(write_fn(new_struct, plddt, True))

    # All confidence metrics from best model
    best_idx = next(i for i, r in rank.items() if r == 0)
    metrics = {}

    scalar_keys = ["confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
                   "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"]
    for k in scalar_keys:
        metrics[k] = round(pred[k][best_idx].item(), 6) if k in pred else 0.0

    # Per-chain-pair iPTM (nested dict, natural in JSON)
    if "pair_chains_iptm" in pred:
        pci = pred["pair_chains_iptm"]
        metrics["pair_chains_iptm"] = {
            i: {j: round(pci[i][j][best_idx].item(), 6) for j in pci[i]}
            for i in pci
        }

    # Optional large outputs
    if write_pae and "pae" in pred:
        np.savez_compressed(out_dir / f"{record.id}_pae.npz", pae=pred["pae"][best_idx].cpu().numpy())
    if write_pde and "pde" in pred:
        np.savez_compressed(out_dir / f"{record.id}_pde.npz", pde=pred["pde"][best_idx].cpu().numpy())
    if write_embeddings and "s" in pred and "z" in pred:
        np.savez_compressed(out_dir / f"{record.id}_embeddings.npz",
                          s=pred["s"].cpu().numpy(), z=pred["z"].cpu().numpy())

    return metrics, best_struct


def _save_results(results: list[dict], path: Path) -> None:
    """Atomic JSON write (write tmp, rename)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.rename(path)


@click.group()
def cli(): pass


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("--out_dir", default="./")
@click.option("--cache", default=lambda: os.environ.get("BOLTZ_CACHE", str(Path("~/.boltz").expanduser())))
@click.option("--checkpoint", type=click.Path(exists=True), default=None)
@click.option("--accelerator", type=click.Choice(["gpu", "cpu", "tenstorrent"]), default="tenstorrent")
@click.option("--recycling_steps", default=3, type=int)
@click.option("--sampling_steps", default=200, type=int)
@click.option("--diffusion_samples", default=1, type=int)
@click.option("--max_parallel_samples", default=5, type=int)
@click.option("--step_scale", default=None, type=float)
@click.option("--output_format", type=click.Choice(["pdb", "cif"]), default="cif")
@click.option("--override", is_flag=True)
@click.option("--seed", default=None, type=int)
@click.option("--use_msa_server", is_flag=True)
@click.option("--msa_server_url", default="https://api.colabfold.com")
@click.option("--msa_pairing_strategy", default="greedy")
@click.option("--msa_server_username", default=None)
@click.option("--msa_server_password", default=None)
@click.option("--api_key_value", default=None)
@click.option("--use_potentials", is_flag=True)
@click.option("--method", default=None)
@click.option("--max_msa_seqs", default=8192, type=int)
@click.option("--subsample_msa", is_flag=True)
@click.option("--num_subsampled_msa", default=1024, type=int)
@click.option("--no_kernels", is_flag=True)
@click.option("--trace", is_flag=True)
@click.option("--write_pae", is_flag=True, help="Write PAE matrix per target")
@click.option("--write_pde", is_flag=True, help="Write PDE matrix per target")
@click.option("--write_embeddings", is_flag=True, help="Write s/z embeddings per target")
@click.option("--affinity_mw_correction", is_flag=True)
@click.option("--sampling_steps_affinity", default=200, type=int)
@click.option("--diffusion_samples_affinity", default=5, type=int)
@click.option("--affinity_checkpoint", type=click.Path(exists=True), default=None)
def predict(data, out_dir, cache, checkpoint, accelerator, recycling_steps, sampling_steps,
            diffusion_samples, max_parallel_samples, step_scale, output_format, override,
            seed, use_msa_server, msa_server_url, msa_pairing_strategy,
            msa_server_username, msa_server_password, api_key_value, use_potentials,
            method, max_msa_seqs, subsample_msa, num_subsampled_msa, no_kernels, trace,
            write_pae, write_pde, write_embeddings, affinity_mw_correction,
            sampling_steps_affinity, diffusion_samples_affinity, affinity_checkpoint):
    """Run Boltz-2 structure prediction.

    DATA is a YAML/FASTA file or a directory of them.
    Model stays in memory across all predictions. Resume by re-running (skips existing outputs).

    \b
    Output:
        boltz_results_<name>/
            msa/            # cached MSA CSVs
            structures/     # one CIF per complex (pLDDT in B-factors)
            results.json    # all confidence metrics + affinity
    """
    use_tt = accelerator == "tenstorrent"
    if use_tt: accelerator = "cpu"

    warnings.filterwarnings("ignore", ".*Tensor Cores.*")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    os.environ.setdefault("CUEQ_DEFAULT_CONFIG", "1")
    os.environ.setdefault("CUEQ_DISABLE_AOT_TUNING", "1")

    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_all(cache)

    if use_msa_server:
        msa_server_username = msa_server_username or os.environ.get("BOLTZ_MSA_USERNAME")
        msa_server_password = msa_server_password or os.environ.get("BOLTZ_MSA_PASSWORD")
        api_key_value = api_key_value or os.environ.get("MSA_API_KEY_VALUE")

    data = Path(data).expanduser()
    out = Path(out_dir).expanduser() / f"boltz_results_{data.stem}"
    msa_dir = out / "msa"
    struct_dir = out / "structures"
    msa_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)
    mol_dir = cache / "mols"

    files = sorted(p for p in (data.glob("*") if data.is_dir() else [data])
                   if p.suffix.lower() in (".fa", ".fas", ".fasta", ".yml", ".yaml"))
    if not files:
        click.echo("No input files found")
        return

    if method and method.lower() not in const.method_types_ids:
        raise ValueError(f"Unknown method: {method}")

    if not override:
        files = [f for f in files if not (struct_dir / f"{f.stem}.{output_format}").exists()]
    if not files:
        click.echo("All predictions complete")
        return

    device = torch.device("cuda:0" if accelerator == "gpu" and torch.cuda.is_available() else "cpu")

    click.echo(f"Predicting {len(files)} structures")
    tokenizer, featurizer = Boltz2Tokenizer(), Boltz2Featurizer()
    ccd = load_canonicals(mol_dir)

    prepare = partial(prepare_features,
        ccd=ccd, mol_dir=mol_dir, msa_dir=msa_dir, tokenizer=tokenizer, featurizer=featurizer,
        use_msa=use_msa_server, msa_url=msa_server_url, msa_strategy=msa_pairing_strategy,
        msa_user=msa_server_username, msa_pass=msa_server_password, api_key=api_key_value,
        max_msa=max_msa_seqs)

    model = Boltz2.load_from_checkpoint(
        checkpoint or cache / "boltz2_conf.ckpt",
        predict_args={"recycling_steps": recycling_steps, "sampling_steps": sampling_steps,
                     "diffusion_samples": diffusion_samples, "max_parallel_samples": max_parallel_samples},
        diffusion_process_args={"step_scale": step_scale or 1.5, "gamma_0": 0.8, "gamma_min": 1.0,
                                "noise_scale": 1.003, "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0,
                                "sigma_data": 16.0, "P_mean": -1.2, "P_std": 1.5,
                                "coordinate_augmentation": True, "alignment_reverse_diff": True,
                                "synchronize_sigmas": True},
        pairformer_args={"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True},
        msa_args={"subsample_msa": subsample_msa, "num_subsampled_msa": num_subsampled_msa, "use_paired_feature": True},
        steering_args={"fk_steering": use_potentials, "physical_guidance_update": use_potentials,
                      "contact_guidance_update": True, "num_particles": 3, "fk_lambda": 4.0,
                      "fk_resampling_interval": 3, "num_gd_steps": 20},
        use_kernels=not no_kernels, use_tenstorrent=use_tt, trace=trace,
    ).eval().to(device)

    # Results — JSON array, rewritten after each prediction for crash-safety
    results_path = out / "results.json"
    results = [] if override or not results_path.exists() else json.loads(results_path.read_text())

    affinity_queue = []
    failed = 0

    for path in tqdm(files, desc="Predicting"):
        t0 = time.time()
        row = {"id": path.stem, "status": "failed"}
        try:
            feats, input_struct = prepare(path, method=method)
            batch_data = to_batch(feats, device)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model.predict_step(batch_data)

            metrics, best_struct = write_result(pred, batch_data, input_struct, struct_dir,
                                                output_format, write_pae, write_pde, write_embeddings)
            if metrics:
                row.update(metrics)
                row["status"] = "ok"
                row["runtime_s"] = round(time.time() - t0, 1)
                if feats["record"].affinity and best_struct is not None:
                    affinity_queue.append((path, best_struct))
            else:
                row["error"] = "prediction exception"
                failed += 1
        except Exception as e:
            traceback.print_exc()
            row["error"] = str(e)[:200]
            failed += 1

        results.append(row)
        _save_results(results, results_path)

    click.echo(f"Structure prediction: {len(files) - failed} ok, {failed} failed")

    # Affinity pass (separate model)
    if affinity_queue:
        click.echo(f"Predicting affinity for {len(affinity_queue)} targets")

        aff_model = Boltz2.load_from_checkpoint(
            affinity_checkpoint or cache / "boltz2_aff.ckpt",
            predict_args={"recycling_steps": 5, "sampling_steps": sampling_steps_affinity,
                         "diffusion_samples": diffusion_samples_affinity, "max_parallel_samples": 1},
            diffusion_process_args={"step_scale": step_scale or 1.5, "gamma_0": 0.8, "gamma_min": 1.0,
                                    "noise_scale": 1.003, "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0,
                                    "sigma_data": 16.0, "P_mean": -1.2, "P_std": 1.5,
                                    "coordinate_augmentation": True, "alignment_reverse_diff": True,
                                    "synchronize_sigmas": True},
            pairformer_args={"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True},
            msa_args={"subsample_msa": subsample_msa, "num_subsampled_msa": num_subsampled_msa, "use_paired_feature": True},
            steering_args={"fk_steering": False, "physical_guidance_update": False, "contact_guidance_update": False,
                          "num_particles": 3, "fk_lambda": 4.0, "fk_resampling_interval": 3, "num_gd_steps": 20},
            affinity_mw_correction=affinity_mw_correction, use_tenstorrent=use_tt, trace=trace,
        ).eval().to(device)

        results_by_id = {r["id"]: r for r in results}
        aff_keys = ["affinity_pred_value", "affinity_probability_binary",
                    "affinity_pred_value1", "affinity_probability_binary1",
                    "affinity_pred_value2", "affinity_probability_binary2"]

        for path, pred_struct in tqdm(affinity_queue, desc="Affinity"):
            try:
                feats, _ = prepare(path, method="other", affinity=True, pred_structure=pred_struct)
                batch_data = to_batch(feats, device)
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    pred = aff_model.predict_step(batch_data)

                if not pred["exception"]:
                    rid = path.stem
                    if rid in results_by_id:
                        for ak in aff_keys:
                            if ak in pred:
                                results_by_id[rid][ak] = round(pred[ak].item(), 6)
            except Exception as e:
                traceback.print_exc()
                click.echo(f"Affinity failed {path.stem}: {e}")

        _save_results(results, results_path)

    click.echo(f"Done. Results: {results_path}")


@cli.command()
@click.option("--max_seq", default=1024, type=int, help="Maximum sequence length to warm up")
@click.option("--max_msa", default=16384, type=int, help="Maximum MSA depth to warm up")
@click.option("--n_samples", default=1, type=int, help="Diffusion batch (multiplicity)")
@click.option("--cache", default=lambda: os.environ.get("BOLTZ_CACHE", str(Path("~/.boltz").expanduser())))
def warmup(max_seq, max_msa, n_samples, cache):
    """Pre-compile all ttnn kernels for Boltz-2 inference."""
    import gc

    from tt_boltz.tenstorrent import (
        filter_dict, PairformerModule, MSAModule, DiffusionModule,
        PAIRFORMER_PAD_MULTIPLE as SEQ_PAD, MSA_PAD_MULTIPLE as MSA_PAD,
        MAX_ATOMS_PER_TOKEN,
    )
    from tt_boltz.boltz2 import get_indexing_matrix

    torch.set_grad_enabled(False)

    seq_bk = list(range(SEQ_PAD, max_seq + 1, SEQ_PAD))
    msa_bk = list(range(MSA_PAD, max_msa + 1, MSA_PAD))

    click.echo(f"seq  buckets ({SEQ_PAD}): {seq_bk}")
    click.echo(f"msa  buckets ({MSA_PAD}): {msa_bk}")
    click.echo("Loading checkpoint …")

    state = torch.load(
        Path(cache) / "boltz2_conf.ckpt",
        map_location="cpu", mmap=True, weights_only=False,
    )["state_dict"]

    total = time.time()

    click.echo(f"\n[1/4] Pairformer (z=128) — {len(seq_bk)} buckets")
    pf = PairformerModule(1, 32, 4, 24, 16, True)
    pf.load_state_dict(filter_dict(state, "pairformer_module"), strict=False)
    for seq in seq_bk:
        t = time.time()
        pf.reset_static_cache()
        actual = seq - 1
        pf(torch.randn(1, actual, 384),
           torch.randn(1, actual, actual, 128),
           mask=torch.ones(1, actual))
        click.echo(f"  seq={actual:>5}→{seq:>5}  {time.time()-t:5.1f}s")
    del pf; gc.collect()

    click.echo(f"\n[2/4] Template Pairformer (z=64) — {len(seq_bk)} buckets")
    pf_tpl = PairformerModule(1, 32, 4, None, None, False)
    pf_tpl.load_state_dict(filter_dict(state, "template_module.pairformer"), strict=False)
    for seq in seq_bk:
        t = time.time()
        pf_tpl.reset_static_cache()
        actual = seq - 1
        pf_tpl(None, torch.randn(1, actual, actual, 64))
        click.echo(f"  seq={actual:>5}→{seq:>5}  {time.time()-t:5.1f}s")
    del pf_tpl; gc.collect()

    n = len(seq_bk) * len(msa_bk)
    click.echo(f"\n[3/4] MSA — {n} combos ({len(seq_bk)} seq × {len(msa_bk)} msa)")
    msa_mod = MSAModule(1, 32, 8, 32, 4)
    msa_mod.load_state_dict(filter_dict(state, "msa_module"), strict=False)
    for seq in seq_bk:
        actual_seq = seq - 1
        for n_msa_val in msa_bk:
            actual_msa = n_msa_val - 1
            t = time.time()
            msa_mod.reset_static_cache()
            try:
                feats = {
                    "msa": torch.randint(33, (1, actual_msa, actual_seq)),
                    "has_deletion": torch.zeros(1, actual_msa, actual_seq, dtype=torch.bool),
                    "deletion_value": torch.zeros(1, actual_msa, actual_seq),
                    "msa_paired": torch.zeros(1, actual_msa, actual_seq),
                    "msa_mask": torch.ones(1, actual_msa, actual_seq),
                    "token_pad_mask": torch.ones(1, actual_seq),
                }
                msa_mod(torch.randn(1, actual_seq, actual_seq, 128),
                        torch.ones(1, actual_seq, 384), feats)
                click.echo(f"  seq={actual_seq:>5}→{seq:>5} msa={actual_msa:>5}→{n_msa_val:>5}  {time.time()-t:5.1f}s")
            except Exception as e:
                click.echo(f"  seq={actual_seq:>5}→{seq:>5} msa={actual_msa:>5}→{n_msa_val:>5}  SKIP ({type(e).__name__})")
    del msa_mod; gc.collect()

    B = n_samples
    W, H = 32, 128
    diff_sd = filter_dict(state, "structure_module.score_model")
    click.echo(f"\n[4/4] Diffusion — {len(seq_bk)} buckets (n_samples={B})")
    for seq in seq_bk:
        actual_seq = seq - 1
        N = seq * MAX_ATOMS_PER_TOKEN
        NW = N // W
        t = time.time()
        try:
            diff = DiffusionModule()
            diff.load_state_dict(diff_sd, strict=False)
            diff(
                torch.randn(B, N, 3), torch.randn(B),
                torch.randn(1, actual_seq, 384), torch.randn(1, actual_seq, 384),
                torch.randn(1, N, 128), torch.randn(1, N, 128),
                torch.randn(1, NW, W, H, 12),
                torch.randn(1, actual_seq, actual_seq, 384),
                torch.randn(1, NW, W, H, 12),
                get_indexing_matrix(NW, W, H, "cpu"),
                torch.ones(1, N), torch.ones(1, N, actual_seq),
            )
            click.echo(f"  seq={actual_seq:>5}→{seq:>5} atoms={N:>6}  {time.time()-t:5.1f}s")
        except Exception as e:
            click.echo(f"  seq={actual_seq:>5}→{seq:>5} atoms={N:>6}  SKIP ({type(e).__name__})")
        del diff; gc.collect()

    click.echo(f"\nDone — {time.time()-total:.0f}s total")


if __name__ == "__main__":
    cli()
