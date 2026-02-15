"""Boltz-2 structure prediction CLI."""

import json
import multiprocessing
import os
import pickle
import random
import tarfile
import traceback
import urllib.request
import warnings
from dataclasses import asdict, replace
from functools import partial
from multiprocessing import Pool
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
from tt_boltz.data.types import Coords, Input, Interface, MSA, Manifest, Record, ResidueConstraints, StructureV2
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
    # Download and extract molecules
    tar_path = cache / "mols.tar"
    if not tar_path.exists():
        urllib.request.urlretrieve(URLS["mols"], tar_path)
    if not (cache / "mols").exists():
        with tarfile.open(tar_path) as tar:
            tar.extractall(cache)

    # Download checkpoints
    download(URLS["conf"], cache / "boltz2_conf.ckpt")
    download(URLS["aff"], cache / "boltz2_aff.ckpt")


def compute_msa(seqs: dict[str, str], target_id: str, msa_dir: Path, url: str, strategy: str,
                username: str = None, password: str = None, api_key: str = None) -> None:
    """Generate MSAs for protein sequences."""
    click.echo(f"MSA for {target_id} ({len(seqs)} sequences)")
    headers = {"Content-Type": "application/json", "X-API-Key": api_key} if api_key else None
    seqs_list = list(seqs.values())

    # Generate paired MSAs (only for multiple sequences)
    paired = (run_mmseqs2(seqs_list, msa_dir / f"{target_id}_paired_tmp", use_env=True,
                         use_pairing=True, host_url=url, pairing_strategy=strategy,
                         msa_server_username=username, msa_server_password=password, auth_headers=headers)
             if len(seqs) > 1 else [""] * len(seqs))

    # Generate unpaired MSAs
    unpaired = run_mmseqs2(seqs_list, msa_dir / f"{target_id}_unpaired_tmp", use_env=True,
                          use_pairing=False, host_url=url, pairing_strategy=strategy,
                          msa_server_username=username, msa_server_password=password, auth_headers=headers)

    # Write MSA CSV files
    for i, name in enumerate(seqs):
        paired_seqs = [s for s in paired[i].strip().splitlines()[1::2][:const.max_paired_seqs] if s != "-" * len(s)]
        unpaired_seqs = unpaired[i].strip().splitlines()[1::2][:const.max_msa_seqs - len(paired_seqs)]
        if paired_seqs:
            unpaired_seqs = unpaired_seqs[1:]  # Skip query if paired exists

        keys = list(range(len(paired_seqs))) + [-1] * len(unpaired_seqs)
        lines = ["key,sequence"] + [f"{k},{s}" for k, s in zip(keys, paired_seqs + unpaired_seqs)]
        (msa_dir / f"{name}.csv").write_text("\n".join(lines))


def process_input(path: Path, ccd: dict, dirs: dict, use_msa: bool, msa_url: str, msa_strategy: str,
                  msa_user: str, msa_pass: str, api_key: str, max_msa: int) -> None:
    """Parse and process a single input file."""
    try:
        # Parse input file
        suffix = path.suffix.lower()
        if suffix in (".fa", ".fas", ".fasta"):
            target = parse_fasta(path, ccd, dirs["mol"], True)
        elif suffix in (".yml", ".yaml"):
            target = parse_yaml(path, ccd, dirs["mol"], True)
        else:
            return

        tid = target.record.id

        # Identify sequences needing MSA generation
        to_gen = {}
        for chain in target.record.chains:
            if chain.mol_type == const.chain_type_ids["PROTEIN"] and chain.msa_id == 0:
                msa_name = f"{tid}_{chain.entity_id}"
                to_gen[msa_name] = target.sequences[chain.entity_id]
                chain.msa_id = dirs["msa_raw"] / f"{msa_name}.csv"
            elif chain.msa_id == 0:
                chain.msa_id = -1

        # Generate MSAs if needed
        if to_gen:
            if not use_msa:
                raise RuntimeError("Missing MSAs, use --use_msa_server")
            compute_msa(to_gen, tid, dirs["msa_raw"], msa_url, msa_strategy, msa_user, msa_pass, api_key)

        # Process and store MSAs
        msa_map = {}
        for i, msa_id in enumerate(sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})):
            msa_path = Path(msa_id)
            out_path = dirs["msa"] / f"{tid}_{i}.npz"
            msa_map[msa_id] = f"{tid}_{i}"

            if not out_path.exists():
                msa = parse_a3m(msa_path, None, max_msa) if msa_path.suffix == ".a3m" else parse_csv(msa_path, max_msa)
                msa.dump(out_path)

        # Update chain MSA IDs
        for chain in target.record.chains:
            if chain.msa_id in msa_map:
                chain.msa_id = msa_map[chain.msa_id]

        # Save all processed data
        for name, template in target.templates.items():
            template.dump(dirs["templates"] / f"{tid}_{name}.npz")
        target.residue_constraints.dump(dirs["constraints"] / f"{tid}.npz")

        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        with (dirs["mols"] / f"{tid}.pkl").open("wb") as f:
            pickle.dump(target.extra_mols, f)

        target.structure.dump(dirs["structures"] / f"{tid}.npz")
        target.record.dump(dirs["records"] / f"{tid}.json")

    except Exception as e:
        traceback.print_exc()
        print(f"Failed: {path}: {e}")


def process_all(files: list[Path], out: Path, mol_dir: Path, use_msa: bool, msa_url: str,
                msa_strategy: str, msa_user: str, msa_pass: str, api_key: str, max_msa: int, threads: int) -> None:
    """Process all input files in parallel."""
    # Setup output directories
    dirs = {
        "msa_raw": out / "msa",
        "records": out / "processed/records",
        "structures": out / "processed/structures",
        "msa": out / "processed/msa",
        "constraints": out / "processed/constraints",
        "templates": out / "processed/templates",
        "mols": out / "processed/mols",
        "mol": mol_dir,
    }
    for d in [*dirs.values(), out / "predictions"]:
        d.mkdir(parents=True, exist_ok=True)

    # Filter out already processed files
    if dirs["records"].exists():
        existing = {Record.load(p).id for p in dirs["records"].glob("*.json")}
        files = [f for f in files if f.stem not in existing]
        if not files:
            click.echo("All processed")
            return

    # Process files
    if files:
        ccd = load_canonicals(mol_dir)
        process_fn = partial(process_input, ccd=ccd, dirs=dirs, use_msa=use_msa, msa_url=msa_url,
                           msa_strategy=msa_strategy, msa_user=msa_user, msa_pass=msa_pass,
                           api_key=api_key, max_msa=max_msa)

        if threads > 1 and len(files) > 1:
            with Pool(min(threads, len(files))) as pool:
                list(tqdm(pool.imap(process_fn, files), total=len(files)))
        else:
            for file in tqdm(files):
                process_fn(file)

    # Create manifest
    records = [Record.load(p) for p in dirs["records"].glob("*.json")]
    Manifest(records).dump(out / "processed/manifest.json")


def load_features(record: Record, dirs: dict, mol_dir: Path, canonicals: dict, tokenizer, featurizer,
                  method: str = None, affinity: bool = False):
    """Load and featurize a single record for prediction."""
    # Load structure
    struct_path = (dirs["pred"] / record.id / f"pre_affinity_{record.id}.npz" if affinity
                  else dirs["structures"] / f"{record.id}.npz")
    struct = StructureV2.load(struct_path)

    # Load MSAs, templates, constraints
    msas = {c.chain_id: MSA.load(dirs["msa"] / f"{c.msa_id}.npz")
           for c in record.chains if c.msa_id != -1}
    templates = ({t.name: StructureV2.load(dirs["templates"] / f"{record.id}_{t.name}.npz")
                 for t in (record.templates or [])} if dirs["templates"].exists() else None)
    constraints = (ResidueConstraints.load(dirs["constraints"] / f"{record.id}.npz")
                  if dirs["constraints"].exists() else None)

    # Load extra molecules
    mol_path = dirs["mols"] / f"{record.id}.pkl"
    extra_mols = pickle.load(mol_path.open("rb")) if mol_path.exists() else {}

    # Tokenize input
    inp = Input(struct, msas, record=record, residue_constraints=constraints,
               templates=templates, extra_mols=extra_mols)
    tok = tokenizer.tokenize(inp)
    
    # Crop tokens for affinity prediction
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

            # Get neighborhood tokens
            if len(chain_tokens) <= 10:
                neighbors = chain_tokens
            else:
                res_window = chain_tokens[(chain_tokens["res_idx"] >= token["res_idx"] - 10) &
                                         (chain_tokens["res_idx"] <= token["res_idx"] + 10)]
                neighbors = res_window[res_window["res_idx"] == token["res_idx"]]
                # Expand until we have 10 tokens
                mi = ma = token["res_idx"]
                while neighbors.size < 10:
                    mi -= 1
                    ma += 1
                    neighbors = res_window[(res_window["res_idx"] >= mi) & (res_window["res_idx"] <= ma)]

            new_ids = set(neighbors["token_idx"]) - cropped
            new_atoms = np.sum(td[list(new_ids)]["atom_num"])

            # Check capacity limits
            if (len(new_ids) > 256 - len(cropped) or
                atoms + new_atoms > 2048 or
                len(prot | new_ids - lig_ids) > 200):
                break

            cropped.update(new_ids)
            atoms += new_atoms
            prot.update(new_ids - lig_ids)

        # Filter tokens and bonds
        td = td[sorted(cropped)]
        tb = tb[np.isin(tb["token_1"], td["token_idx"]) & np.isin(tb["token_2"], td["token_idx"])]
        tok = replace(tok, tokens=td, bonds=tb)

    # Load all required molecules
    mols = {**canonicals, **extra_mols}
    needed = set(tok.tokens["res_name"].tolist()) - set(mols)
    mols.update(load_molecules(mol_dir, needed))

    # Get constraints from inference options
    opts = record.inference_options
    pocket, contact = ((opts.pocket_constraints, opts.contact_constraints) if opts else (None, None))

    # Featurize
    feats = featurizer.process(
        tok, np.random.default_rng(42), mols, False, const.max_msa_seqs,
        pad_to_max_seqs=False, single_sequence_prop=0.0, compute_frames=True,
        inference_pocket_constraints=pocket, inference_contact_constraints=contact,
        compute_constraint_features=True, override_method=method, compute_affinity=affinity
    )
    feats["record"] = record
    return feats


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


def write_structure(pred: dict, batch: dict, data_dir: Path, out_dir: Path, fmt: str, embeddings: bool) -> bool:
    """Write predicted structure and confidence metrics."""
    if pred["exception"]:
        return True

    record = batch["record"][0]
    struct = StructureV2.load(data_dir / f"{record.id}.npz").remove_invalid_chains()

    # Rank models by confidence
    confidence = pred.get("confidence_score", torch.zeros(1))
    rank = {i.item(): r for r, i in enumerate(torch.argsort(confidence, descending=True))}

    # Write each model
    mask_1d = pred["masks"].squeeze(0) if pred["masks"].dim() > 1 else pred["masks"]
    out_path = out_dir / record.id
    out_path.mkdir(exist_ok=True)

    for model_idx in range(pred["coords"].shape[0]):
        coords = pred["coords"][model_idx][mask_1d.bool()].cpu().numpy()

        # Update structure with predicted coordinates
        atoms, residues = struct.atoms.copy(), struct.residues.copy()
        atoms["coords"], atoms["is_present"] = coords, True
        residues["is_present"] = True
        new_struct = replace(struct, atoms=atoms, residues=residues,
                            interfaces=np.array([], dtype=Interface),
                            coords=np.array([(x,) for x in coords], dtype=Coords))

        # Write structure file
        model_name = f"{record.id}_model_{rank.get(model_idx, model_idx)}"
        write_fn = to_pdb if fmt == "pdb" else to_mmcif
        (out_path / f"{model_name}.{fmt}").write_text(write_fn(new_struct, pred.get("plddt", [None] * (model_idx + 1))[model_idx], True))

        # Save affinity structure
        if record.affinity and rank.get(model_idx, model_idx) == 0:
            np.savez_compressed(out_path / f"pre_affinity_{record.id}.npz", **asdict(new_struct))

        # Write confidence metrics
        plddt = pred.get("plddt", [None] * (model_idx + 1))[model_idx]
        if plddt is not None:
            conf_keys = ["confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
                        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"]
            conf = {k: pred[k][model_idx].item() for k in conf_keys}
            conf["chains_ptm"] = {i: pred["pair_chains_iptm"][i][i][model_idx].item()
                                 for i in pred["pair_chains_iptm"]}
            conf["pair_chains_iptm"] = {i: {j: pred["pair_chains_iptm"][i][j][model_idx].item()
                                            for j in pred["pair_chains_iptm"][i]}
                                       for i in pred["pair_chains_iptm"]}

            (out_path / f"confidence_{model_name}.json").write_text(json.dumps(conf, indent=4))
            np.savez_compressed(out_path / f"plddt_{model_name}.npz", plddt=plddt.cpu().numpy())

        # Write PAE and PDE if available
        if "pae" in pred:
            np.savez_compressed(out_path / f"pae_{model_name}.npz", pae=pred["pae"][model_idx].cpu().numpy())
        if "pde" in pred:
            np.savez_compressed(out_path / f"pde_{model_name}.npz", pde=pred["pde"][model_idx].cpu().numpy())

    # Write embeddings
    if embeddings and "s" in pred and "z" in pred:
        np.savez_compressed(out_path / f"embeddings_{record.id}.npz",
                          s=pred["s"].cpu().numpy(), z=pred["z"].cpu().numpy())
    return False


def write_affinity(pred: dict, batch: dict, out_dir: Path) -> bool:
    """Write affinity prediction results."""
    if pred["exception"]:
        return True

    record_id = batch["record"][0].id
    affinity = {
        "affinity_pred_value": pred["affinity_pred_value"].item(),
        "affinity_probability_binary": pred["affinity_probability_binary"].item(),
    }

    # Add ensemble predictions if available
    if "affinity_pred_value1" in pred:
        affinity.update({f"affinity_pred_value{i}": pred[f"affinity_pred_value{i}"].item() for i in [1, 2]})
        affinity.update({f"affinity_probability_binary{i}": pred[f"affinity_probability_binary{i}"].item() for i in [1, 2]})

    out_path = out_dir / record_id
    out_path.mkdir(exist_ok=True)
    (out_path / f"affinity_{record_id}.json").write_text(json.dumps(affinity, indent=4))
    return False


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
@click.option("--write_full_pae", is_flag=True)
@click.option("--write_full_pde", is_flag=True)
@click.option("--output_format", type=click.Choice(["pdb", "cif"]), default="cif")
@click.option("--override", is_flag=True)
@click.option("--seed", default=None, type=int)
@click.option("--use_msa_server", is_flag=True)
@click.option("--msa_server_url", default="https://api.colabfold.com")
@click.option("--msa_pairing_strategy", default="greedy")
@click.option("--msa_server_username", default=None)
@click.option("--msa_server_password", default=None)
@click.option("--api_key_header", default=None)
@click.option("--api_key_value", default=None)
@click.option("--use_potentials", is_flag=True)
@click.option("--method", default=None)
@click.option("--preprocessing_threads", default=multiprocessing.cpu_count(), type=int)
@click.option("--affinity_mw_correction", is_flag=True)
@click.option("--sampling_steps_affinity", default=200, type=int)
@click.option("--diffusion_samples_affinity", default=5, type=int)
@click.option("--affinity_checkpoint", type=click.Path(exists=True), default=None)
@click.option("--max_msa_seqs", default=8192, type=int)
@click.option("--subsample_msa", is_flag=True)
@click.option("--num_subsampled_msa", default=1024, type=int)
@click.option("--no_kernels", is_flag=True)
@click.option("--write_embeddings", is_flag=True)
@click.option("--trace", is_flag=True)
@click.option("--num_workers", default=2, type=int, hidden=True)  # kept for compatibility but ignored
def predict(data, out_dir, cache, checkpoint, accelerator, recycling_steps, sampling_steps,
            diffusion_samples, max_parallel_samples, step_scale, write_full_pae, write_full_pde,
            output_format, override, seed, use_msa_server, msa_server_url, msa_pairing_strategy,
            msa_server_username, msa_server_password, api_key_header, api_key_value, use_potentials,
            method, preprocessing_threads, affinity_mw_correction, sampling_steps_affinity,
            diffusion_samples_affinity, affinity_checkpoint, max_msa_seqs, subsample_msa,
            num_subsampled_msa, no_kernels, write_embeddings, trace, num_workers):
    """Run Boltz-2 structure prediction."""
    # Normalize output format: mmcif -> cif
    if output_format == "mmcif":
        output_format = "cif"
    
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
    out.mkdir(parents=True, exist_ok=True)
    mol_dir = cache / "mols"
    
    files = [p for p in (data.glob("*") if data.is_dir() else [data]) 
             if p.suffix.lower() in (".fa", ".fas", ".fasta", ".yml", ".yaml")]
    
    if method and method.lower() not in const.method_types_ids:
        raise ValueError(f"Unknown method: {method}")
    
    process_all(files, out, mol_dir, use_msa_server, msa_server_url, msa_pairing_strategy,
                msa_server_username, msa_server_password, api_key_value, max_msa_seqs, preprocessing_threads)
    
    manifest = Manifest.load(out / "processed/manifest.json")
    pred_dir = out / "predictions"
    
    dirs = {"structures": out / "processed/structures", "msa": out / "processed/msa",
            "constraints": out / "processed/constraints", "templates": out / "processed/templates",
            "mols": out / "processed/mols", "pred": pred_dir}
    
    device = torch.device("cuda:0" if accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    
    # Structure prediction
    existing = {d.name for d in pred_dir.iterdir() if d.is_dir()} if pred_dir.exists() else set()
    records = [r for r in manifest.records if override or r.id not in existing]
    
    if records:
        click.echo(f"Predicting {len(records)} structures")
        tokenizer, featurizer = Boltz2Tokenizer(), Boltz2Featurizer()
        canonicals = load_canonicals(mol_dir)
        
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
        
        failed = 0
        for record in tqdm(records, desc="Predicting"):
            try:
                feats = load_features(record, dirs, mol_dir, canonicals, tokenizer, featurizer, method)
                batch = to_batch(feats, device)
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    pred = model.predict_step(batch)
                if write_structure(pred, batch, dirs["structures"], pred_dir, output_format, write_embeddings):
                    failed += 1
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"Failed {record.id}: {e}")
                failed += 1
        print(f"Failed: {failed}")
    
    # Affinity prediction
    aff_records = [r for r in manifest.records if r.affinity and 
                   (override or not (pred_dir / r.id / f"affinity_{r.id}.json").exists())]
    
    if aff_records:
        click.echo(f"Predicting affinity for {len(aff_records)}")
        tokenizer, featurizer = Boltz2Tokenizer(), Boltz2Featurizer()
        canonicals = load_canonicals(mol_dir)
        
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
        
        failed = 0
        for record in tqdm(aff_records, desc="Affinity"):
            try:
                feats = load_features(record, dirs, mol_dir, canonicals, tokenizer, featurizer, "other", affinity=True)
                batch = to_batch(feats, device)
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    pred = aff_model.predict_step(batch)
                if write_affinity(pred, batch, pred_dir):
                    failed += 1
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"Failed affinity {record.id}: {e}")
                failed += 1
        print(f"Affinity failed: {failed}")


@cli.command()
@click.option("--max_seq", default=1024, type=int, help="Maximum sequence length to warm up")
@click.option("--max_msa", default=16384, type=int, help="Maximum MSA depth to warm up (const.max_msa_seqs)")
@click.option("--n_samples", default=1, type=int, help="Diffusion batch (multiplicity)")
@click.option("--cache", default=lambda: os.environ.get("BOLTZ_CACHE", str(Path("~/.boltz").expanduser())))
def warmup(max_seq, max_msa, n_samples, cache):
    """Pre-compile all ttnn kernels for Boltz-2 inference.

    Exercises each TT module at every bucketed input shape so the device
    program cache is fully populated before real predictions begin.

    Each module uses a single block/layer since all blocks within a module
    share identical kernel shapes — one pass compiles the full set.
    """
    import gc
    import time

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

    # ── 1. Pairformer (z=128, with s) ─────────────────────────────────────
    # Covers trunk (64 blocks) + confidence (8 blocks).  Also pre-caches
    # TriAttn/TriMult/Transition kernels reused by MSA's PairformerLayer.
    # seq-1 forces padding → mask ops (multiply_ in TriMult, add in TriAttn)
    # are compiled.  1 block is sufficient: all blocks share kernel shapes.
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

    # ── 2. Template Pairformer (z=64, no s) ───────────────────────────────
    # z_dim=64 gives different weight shapes → separate compilation needed.
    # No AttentionPairBias (transform_s=False).  1 block sufficient.
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

    # ── 3. MSA ────────────────────────────────────────────────────────────
    # PairWeightedAveraging and OuterProductMean matmul shapes depend on
    # both seq_len AND n_msa → full grid required.  1 layer sufficient:
    # all MSA layers share kernel shapes.  TriAttn/TriMult kernels inside
    # MSA's PairformerLayer are cache hits from step 1.
    n = len(seq_bk) * len(msa_bk)
    click.echo(f"\n[3/4] MSA — {n} combos ({len(seq_bk)} seq × {len(msa_bk)} msa)")
    msa = MSAModule(1, 32, 8, 32, 4)
    msa.load_state_dict(filter_dict(state, "msa_module"), strict=False)
    for seq in seq_bk:
        actual_seq = seq - 1
        for n_msa_val in msa_bk:
            actual_msa = n_msa_val - 1
            t = time.time()
            msa.reset_static_cache()
            try:
                feats = {
                    "msa": torch.randint(33, (1, actual_msa, actual_seq)),
                    "has_deletion": torch.zeros(1, actual_msa, actual_seq, dtype=torch.bool),
                    "deletion_value": torch.zeros(1, actual_msa, actual_seq),
                    "msa_paired": torch.zeros(1, actual_msa, actual_seq),
                    "msa_mask": torch.ones(1, actual_msa, actual_seq),
                    "token_pad_mask": torch.ones(1, actual_seq),
                }
                msa(torch.randn(1, actual_seq, actual_seq, 128),
                    torch.ones(1, actual_seq, 384), feats)
                click.echo(f"  seq={actual_seq:>5}→{seq:>5} msa={actual_msa:>5}→{n_msa_val:>5}  {time.time()-t:5.1f}s")
            except Exception as e:
                click.echo(f"  seq={actual_seq:>5}→{seq:>5} msa={actual_msa:>5}→{n_msa_val:>5}  SKIP ({type(e).__name__})")
    del msa; gc.collect()

    # ── 4. Diffusion ──────────────────────────────────────────────────────
    # Atom count = padded_seq × MAX_ATOMS_PER_TOKEN.  Encoder/decoder each
    # have 3 layers that split bias dim 12→4 per layer — layer count cannot
    # be reduced (different per-layer shapes).  Fresh module per bucket:
    # ttnn reshape/permute can mutate cached tensor metadata.
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
                torch.randn(B, N, 3),
                torch.randn(B),
                torch.randn(1, actual_seq, 384),
                torch.randn(1, actual_seq, 384),
                torch.randn(1, N, 128),
                torch.randn(1, N, 128),
                torch.randn(1, NW, W, H, 12),
                torch.randn(1, actual_seq, actual_seq, 384),
                torch.randn(1, NW, W, H, 12),
                get_indexing_matrix(NW, W, H, "cpu"),
                torch.ones(1, N),
                torch.ones(1, N, actual_seq),
            )
            click.echo(f"  seq={actual_seq:>5}→{seq:>5} atoms={N:>6}  {time.time()-t:5.1f}s")
        except Exception as e:
            click.echo(f"  seq={actual_seq:>5}→{seq:>5} atoms={N:>6}  SKIP ({type(e).__name__})")
        del diff; gc.collect()

    click.echo(f"\nDone — {time.time()-total:.0f}s total")


if __name__ == "__main__":
    cli()
