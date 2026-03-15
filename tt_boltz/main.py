"""Boltz-2 structure prediction CLI."""

# Suppress noisy ttnn/loguru output before any import pulls in ttnn.
# These are defaults — --debug mode removes them so everything is visible.
import os as _os, sys as _sys
if "--debug" not in _sys.argv:
    _os.environ.setdefault("LOGURU_LEVEL", "WARNING")
    _os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")

import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import tarfile
import time
import traceback
import urllib.request
import warnings
from queue import Empty
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
from tt_boltz.progress import DebugDisplay, NullDisplay, ProgressDisplay, make_progress_fn

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
    """Generate MSAs for protein sequences via ColabFold server."""
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


_COLABFOLD_SEARCH_PATHS = [
    Path.home() / "localcolabfold" / ".pixi" / "envs" / "default" / "bin" / "colabfold_search",
]


def _find_colabfold_search() -> str:
    """Find colabfold_search binary on PATH or at common install locations."""
    found = shutil.which("colabfold_search")
    if found:
        return found
    for p in _COLABFOLD_SEARCH_PATHS:
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    raise RuntimeError(
        "colabfold_search not found.\n"
        "Install localcolabfold and/or activate the environment that provides it:\n"
        "  https://github.com/YoshitakaMo/localcolabfold"
    )


_MMSEQS_SEARCH_PATHS = [
    Path.home() / "localcolabfold" / ".pixi" / "envs" / "default" / "bin" / "mmseqs",
]


def _find_pixi() -> str | None:
    """Find pixi binary on PATH or common install location."""
    found = shutil.which("pixi")
    if found:
        return found
    p = Path.home() / ".pixi" / "bin" / "pixi"
    if p.is_file() and os.access(p, os.X_OK):
        return str(p)
    return None


def _missing_offline_tools() -> list[str]:
    missing = []
    try:
        _find_mmseqs()
    except Exception:
        missing.append("mmseqs")
    try:
        _find_colabfold_search()
    except Exception:
        missing.append("colabfold_search")
    return missing


def _ensure_pixi() -> str:
    """Ensure pixi is installed and return its path."""
    pixi = _find_pixi()
    if pixi:
        return pixi
    if not shutil.which("curl"):
        raise RuntimeError("curl is required to auto-install pixi")
    click.echo("Installing pixi ...")
    subprocess.run(
        ["bash", "-lc", "curl -fsSL https://pixi.sh/install.sh | sh"],
        check=True,
    )
    pixi = _find_pixi()
    if not pixi:
        raise RuntimeError("pixi install finished but pixi binary was not found")
    return pixi


def _ensure_aria2(pixi: str) -> None:
    """Install aria2 via pixi if not already available."""
    if shutil.which("aria2c"):
        return
    click.echo("Installing aria2 for fast parallel downloads ...")
    subprocess.run([pixi, "global", "install", "aria2"], check=True)


def _ensure_offline_tools(install_tools: bool) -> None:
    """Ensure mmseqs + colabfold_search + aria2 are available; optionally install them."""
    missing = _missing_offline_tools()
    need_aria2 = not shutil.which("aria2c")

    if not missing and not need_aria2:
        return
    if not install_tools:
        all_missing = missing + (["aria2c"] if need_aria2 else [])
        raise RuntimeError(
            "Missing offline MSA tools: " + ", ".join(all_missing) + "\n"
            "Rerun with: tt-boltz msa --install-tools"
        )

    pixi = _ensure_pixi()
    _ensure_aria2(pixi)

    if missing:
        click.echo("Missing offline MSA tools: " + ", ".join(missing))
        click.echo("Installing localcolabfold toolchain ...")

        if not shutil.which("git"):
            raise RuntimeError("git is required to auto-install localcolabfold")

        lc = Path.home() / "localcolabfold"
        if not lc.exists():
            subprocess.run(
                ["git", "clone", "https://github.com/YoshitakaMo/localcolabfold.git", str(lc)],
                check=True,
            )

        subprocess.run([pixi, "install"], cwd=str(lc), check=True)
        subprocess.run([pixi, "run", "setup"], cwd=str(lc), check=True)

        missing = _missing_offline_tools()
        if missing:
            raise RuntimeError(
                "localcolabfold setup completed but tools are still missing: "
                + ", ".join(missing)
            )


def _find_mmseqs() -> str:
    """Find mmseqs binary on PATH or at common install locations."""
    found = shutil.which("mmseqs")
    if found:
        return found
    for p in _MMSEQS_SEARCH_PATHS:
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    raise RuntimeError(
        "mmseqs not found.\n"
        "Install localcolabfold and/or activate the environment that provides it:\n"
        "  https://github.com/YoshitakaMo/localcolabfold"
    )


def _download_file(url: str, dest: Path, max_retries: int = 5) -> None:
    """Download a large file with retries and tool fallback."""
    click.echo(f"  Downloading {dest.name} ...")
    tools = []
    if shutil.which("aria2c"):
        tools.append(("aria2c", [
            "aria2c", "--max-connection-per-server=8", "--split=8",
            "--allow-overwrite=true", "--auto-file-renaming=false",
            "--retry-wait=5", "--max-tries=0",
            "-o", dest.name, "-d", str(dest.parent), url]))
    if shutil.which("curl"):
        tools.append(("curl", [
            "curl", "-L", "--retry", "10", "--retry-delay", "5",
            "-C", "-", "--progress-bar", "-o", str(dest), url]))
    if shutil.which("wget"):
        tools.append(("wget", [
            "wget", "-c", "--tries=10", "--wait=5",
            "-O", str(dest), url]))
    if not tools:
        click.echo("    (no aria2c/curl/wget — using Python urllib, may be slow)")
        urllib.request.urlretrieve(url, dest)
        return
    for attempt in range(1, max_retries + 1):
        for name, cmd in tools:
            try:
                subprocess.run(cmd, check=True)
                return
            except subprocess.CalledProcessError:
                click.echo(f"  {name} failed (attempt {attempt}/{max_retries}), retrying ...")
    raise RuntimeError(f"Download failed after {max_retries} attempts: {url}")


def _mmseqs_index_exists(db_dir: Path, db_name: str) -> bool:
    """Return True if MMseqs index for db_name already exists."""
    return (db_dir / f"{db_name}.idx").exists()


def _validate_offline_msa_db(db_path: Path, require_envdb: bool = False) -> None:
    """Validate local MSA DB layout and required ready markers."""
    db_path = db_path.expanduser()
    if not db_path.exists():
        raise RuntimeError(
            f"Offline MSA DB path does not exist: {db_path}\n"
            "Run: tt-boltz msa --path <path>  (or use --use_msa_server)"
        )
    if not db_path.is_dir():
        raise RuntimeError(f"Offline MSA DB path must be a directory: {db_path}")

    uniref_ready = db_path / "UNIREF30_READY"
    if not uniref_ready.exists():
        raise RuntimeError(
            f"Offline MSA DB is incomplete at {db_path} (missing UNIREF30_READY).\n"
            "Run: tt-boltz msa --db uniref30 --path "
            f"{db_path}  (or use --use_msa_server)"
        )

    if require_envdb and not (db_path / "COLABDB_READY").exists():
        raise RuntimeError(
            f"--use_envdb requested but EnvDB is not set up at {db_path}.\n"
            f"Run: tt-boltz msa --db all --path {db_path}"
        )


def compute_msa_offline(seqs: dict[str, str], target_id: str, msa_dir: Path,
                        db_path: str, use_env: bool = False,
                        pairing_strategy: str = "greedy") -> None:
    """Generate MSAs locally via colabfold_search against a local database."""
    click.echo(f"MSA for {target_id} ({len(seqs)} sequences, offline, pairing={pairing_strategy})")
    colabfold_bin = _find_colabfold_search()
    strategy_map = {"greedy": "0", "complete": "1"}
    strategy_val = strategy_map.get(pairing_strategy, pairing_strategy)
    tmp = msa_dir / f"_offline_tmp_{os.getpid()}"
    tmp.mkdir(exist_ok=True)
    try:
        fasta = tmp / "query.fasta"
        with open(fasta, "w") as f:
            for name, seq in seqs.items():
                f.write(f">{name}\n{seq}\n")
        a3m_out = tmp / "a3m"
        a3m_out.mkdir(exist_ok=True)
        cmd = [colabfold_bin, str(fasta), db_path, str(a3m_out),
               "--use-env", "1" if use_env else "0", "--use-templates", "0",
               "--db-load-mode", "2", "--threads", str(os.cpu_count() or 1)]
        if len(seqs) > 1:
            cmd += ["--pair-mode", "unpaired_paired",
                    "--pairing_strategy", strategy_val]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"colabfold_search failed (exit {result.returncode})")
        for name in seqs:
            src = a3m_out / f"{name}.a3m"
            if src.exists():
                shutil.copy2(src, msa_dir / f"{name}.a3m")
            else:
                click.echo(f"  warning: no A3M for {name}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def prepare_features(path, ccd, mol_dir, msa_dir, tokenizer, featurizer,
                     use_msa, msa_url, msa_strategy, msa_user, msa_pass, api_key,
                     max_msa, msa_db_path=None, use_envdb=False, method=None,
                     affinity=False, pred_structure=None):
    """Parse, resolve MSA, tokenize, featurize — all in memory.

    MSA files are cached in msa_dir by sequence hash — the same
    protein sequence is never searched twice across any input file or run.
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

    # Identify protein chains needing MSA, keyed by sequence hash for global caching
    to_gen = {}
    for chain in record.chains:
        if chain.mol_type == const.chain_type_ids["PROTEIN"] and chain.msa_id == 0:
            seq = target.sequences[chain.entity_id]
            seq_hash = hashlib.sha256(seq.encode()).hexdigest()[:16]
            a3m = msa_dir / f"{seq_hash}.a3m"
            chain.msa_id = str(a3m) if a3m.exists() else str(msa_dir / f"{seq_hash}.csv")
            if not Path(chain.msa_id).exists():
                to_gen[seq_hash] = seq
        elif chain.msa_id == 0:
            chain.msa_id = -1

    if to_gen:
        if msa_db_path:
            compute_msa_offline(to_gen, record.id, msa_dir, msa_db_path,
                                use_env=use_envdb, pairing_strategy=msa_strategy)
        elif use_msa:
            compute_msa(to_gen, record.id, msa_dir, msa_url, msa_strategy, msa_user, msa_pass, api_key)
        else:
            raise RuntimeError(
                "Missing MSAs. Use one of:\n"
                "  1) Online:  --use_msa_server\n"
                "  2) Offline: tt-boltz msa  (then rerun predict)"
            )
        for chain in record.chains:
            if isinstance(chain.msa_id, str) and not Path(chain.msa_id).exists():
                a3m = Path(chain.msa_id).with_suffix(".a3m")
                if a3m.exists():
                    chain.msa_id = str(a3m)

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


def _atomic_write(path: Path, content: str):
    """Write file atomically via tmp+rename to prevent corruption on crash."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.rename(path)


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
            _atomic_write(out_dir / f"{record.id}.{fmt}", write_fn(new_struct, plddt, True))
        else:
            _atomic_write(out_dir / f"{record.id}_model_{model_rank}.{fmt}", write_fn(new_struct, plddt, True))

    best_idx = next(i for i, r in rank.items() if r == 0)
    num_samples = pred["coords"].shape[0]
    metrics = {}

    scalar_keys = ["confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
                   "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"]

    def _scalars(idx):
        return {k: round(pred[k][idx].item(), 6) if k in pred else 0.0 for k in scalar_keys}

    metrics.update(_scalars(best_idx))

    if "pair_chains_iptm" in pred:
        pci = pred["pair_chains_iptm"]
        metrics["pair_chains_iptm"] = {
            i: {j: round(pci[i][j][best_idx].item(), 6) for j in pci[i]}
            for i in pci
        }
        metrics["chains_ptm"] = {
            i: round(pci[i][i][best_idx].item(), 6) for i in pci if i in pci[i]
        }

    if num_samples > 1:
        idx_by_rank = sorted(rank, key=rank.get)
        metrics["all_runs"] = [{"rank": rank[i], **_scalars(i)} for i in idx_by_rank]

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


def _append_result(row: dict, path: Path) -> None:
    """Append one result row to results.json atomically.

    Safe for concurrent workers: reads existing, merges, writes via rename.
    """
    existing = json.loads(path.read_text()) if path.exists() else []
    existing = [r for r in existing if r["id"] != row["id"]]
    existing.append(row)
    _save_results(existing, path)


def _predict_worker(device_id, file_paths, cfg, queue, progress_queue,
                     suppress_output=True):
    """Worker process: run predictions on one TT device.

    Each worker is pinned to a single physical card via TT_VISIBLE_DEVICES.
    All workers share the same on-disk kernel cache (no TT_METAL_CACHE override).
    """
    watchdog_queue = cfg.get("watchdog_queue")

    def _emit(ev: dict):
        try:
            progress_queue.put_nowait(ev)
            if watchdog_queue is not None:
                watchdog_queue.put_nowait(ev)
        except Exception:
            pass

    def _pq(event, **kw):
        _emit({"dev": device_id, "event": event, **kw})

    # Silence all output so it doesn't corrupt the Rich Live display.
    # In subprocesses we redirect both stdout+stderr at the OS fd level
    # (catches C++ library noise too). In the main process we only redirect
    # Python stdout — Rich needs stderr there.
    if suppress_output:
        import sys as _sys
        _devnull = open(os.devnull, "w")
        _sys.stdout = _devnull
        if device_id > 0:  # subprocesses
            _sys.stderr = _devnull
            _dn_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_dn_fd, 1)   # fd 1 = stdout  (C++ prints)
            os.dup2(_dn_fd, 2)   # fd 2 = stderr  (C++ warnings/logs)
            os.close(_dn_fd)

    _pq("init", assigned=len(file_paths))
    results = []
    try:
        os.environ["TT_VISIBLE_DEVICES"] = str(device_id)
        torch_device = torch.device("cpu")
        tokenizer, featurizer = Boltz2Tokenizer(), Boltz2Featurizer()
        ccd = load_canonicals(Path(cfg["mol_dir"]))
        prepare = partial(
            prepare_features, ccd=ccd, mol_dir=Path(cfg["mol_dir"]),
            msa_dir=Path(cfg["msa_dir"]), tokenizer=tokenizer, featurizer=featurizer,
            use_msa=cfg["use_msa_server"], msa_url=cfg["msa_server_url"],
            msa_strategy=cfg["msa_pairing_strategy"], msa_user=cfg["msa_server_username"],
            msa_pass=cfg["msa_server_password"], api_key=cfg["api_key_value"],
            max_msa=cfg["max_msa_seqs"], msa_db_path=cfg.get("msa_db_path"),
            use_envdb=cfg.get("use_envdb", False),
        )

        _pq("loading")
        model = Boltz2.load_from_checkpoint(
            cfg["conf_ckpt"], **cfg["conf_kwargs"],
        ).eval().to(torch_device)
        if watchdog_queue is None:
            model.progress_fn = make_progress_fn(progress_queue, device_id)
        else:
            class _DualQueue:
                def put_nowait(self, ev):
                    _emit(ev)
            model.progress_fn = make_progress_fn(_DualQueue(), device_id)

        affinity_items = []
        struct_dir = Path(cfg["struct_dir"])

        for p in file_paths:
            path = Path(p)
            row = {"id": path.stem, "status": "failed"}
            t0 = time.time()
            _pq("start", name=path.stem)
            try:
                _pq("stage", stage="msa")
                feats, input_struct = prepare(path, method=cfg["method"])
                _pq("stage", stage="saving")
                batch = to_batch(feats, torch_device)
                with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    pred = model.predict_step(batch)
                _pq("stage", stage="saving")
                metrics, best = write_result(
                    pred, batch, input_struct, struct_dir,
                    cfg["output_format"], cfg["write_pae"], cfg["write_pde"], cfg["write_embeddings"],
                )
                if metrics:
                    row.update(metrics)
                    row["status"] = "ok"
                    row["runtime_s"] = round(time.time() - t0, 1)
                    if feats["record"].affinity and best is not None:
                        affinity_items.append((path, best))
            except Exception as e:
                traceback.print_exc()
                row["error"] = str(e)[:80]
            elapsed = round(time.time() - t0, 1)
            _pq("done", name=path.stem, time=elapsed, status=row["status"],
                error=row.get("error", ""))
            results.append(row)
            if "results_path" in cfg:
                try: _append_result(row, Path(cfg["results_path"]))
                except Exception: pass

        if affinity_items:
            aff_model = Boltz2.load_from_checkpoint(
                cfg["aff_ckpt"], **cfg["aff_kwargs"],
            ).eval().to(torch_device)
            rows_by_id = {r["id"]: r for r in results}
            aff_keys = ["affinity_pred_value", "affinity_probability_binary",
                        "affinity_pred_value1", "affinity_probability_binary1",
                        "affinity_pred_value2", "affinity_probability_binary2"]
            for path, pred_struct in affinity_items:
                try:
                    feats, _ = prepare(path, method="other", affinity=True, pred_structure=pred_struct)
                    batch = to_batch(feats, torch_device)
                    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                        pred = aff_model.predict_step(batch)
                    if not pred["exception"] and path.stem in rows_by_id:
                        for ak in aff_keys:
                            if ak in pred:
                                rows_by_id[path.stem][ak] = round(pred[ak].item(), 6)
                except Exception as e:
                    traceback.print_exc()

        failed = sum(1 for r in results if r["status"] == "failed")
        queue.put({"ok": True, "dev": device_id, "results": results, "failed": failed})
    except Exception as e:
        traceback.print_exc()
        queue.put({"ok": False, "dev": device_id, "error": str(e), "results": results, "failed": len(file_paths)})


def _reset_tt_device(device_id: int, retries: int = 2) -> None:
    """Reset one TT device via tt-smi.

    device_id here is the same index used for TT_VISIBLE_DEVICES in workers.
    Use --no_reinit to avoid global board re-init while other workers run.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(["tt-smi", "-r", str(device_id), "--no_reinit"], check=True)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            if attempt < retries:
                time.sleep(2.0)
    raise last_err


@click.group()
def cli(): pass


_MSA_DBS = {
    "uniref30": {
        "url": "https://opendata.mmseqs.org/colabfold/uniref30_2302.db.tar.gz",
        "name": "uniref30_2302_db",
        "ready": "UNIREF30_READY",
    },
    "envdb": {
        "url": "https://opendata.mmseqs.org/colabfold/colabfold_envdb_202108.db.tar.gz",
        "name": "colabfold_envdb_202108_db",
        "ready": "COLABDB_READY",
    },
}


@cli.command()
@click.option("--db", type=click.Choice(["uniref30", "envdb", "all"]), default="uniref30",
              help="Database to download: uniref30 (~500GB), envdb (~800GB), or all (~1.3TB)")
@click.option("--path", default=None, type=click.Path(),
              help="Database location (default: ~/.boltz/msa_db)")
@click.option("--install-tools/--no-install-tools", default=True,
              help="Auto-install missing mmseqs/colabfold_search via localcolabfold")
def msa(db, path, install_tools):
    """Download MSA databases for offline structure prediction.

    \b
    After setup, predictions auto-detect the database:
        tt-boltz msa
        tt-boltz predict input.yaml
    """
    cache = Path(os.environ.get("BOLTZ_CACHE", str(Path("~/.boltz").expanduser())))
    db_dir = Path(path).expanduser() if path else cache / "msa_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    _ensure_offline_tools(install_tools=install_tools)
    mmseqs = _find_mmseqs()
    dbs_to_setup = ["uniref30", "envdb"] if db == "all" else [db]

    for name in dbs_to_setup:
        info = _MSA_DBS[name]
        ready = db_dir / info["ready"]
        if ready.exists():
            click.echo(f"{name}: already set up")
            continue

        click.echo(f"\n{name}: downloading")
        tarball = db_dir / Path(info["url"]).name
        _download_file(info["url"], tarball)

        click.echo(f"{name}: extracting")
        subprocess.run(["tar", "-xzf", str(tarball), "-C", str(db_dir)], check=True)

        if _mmseqs_index_exists(db_dir, info["name"]):
            click.echo(f"{name}: index already present")
        else:
            click.echo(f"{name}: building index (this takes a while)")
            subprocess.run(
                [mmseqs, "createindex", str(db_dir / info["name"]),
                 str(db_dir / f"tmp_{name}"), "--remove-tmp-files", "1"],
                check=True)

        if name == "uniref30":
            tax_url = "https://opendata.mmseqs.org/colabfold/uniref30_2302_newtaxonomy.tar.gz"
            tax_tar = db_dir / "uniref30_2302_newtaxonomy.tar.gz"
            _download_file(tax_url, tax_tar)
            subprocess.run(["tar", "-xzf", str(tax_tar), "-C", str(db_dir)], check=True)
            mapping = db_dir / "uniref30_2302_db_mapping"
            if mapping.exists():
                subprocess.run(
                    [mmseqs, "createbintaxmapping", str(mapping), str(mapping) + ".bin"],
                    check=False)
                bin_path = Path(str(mapping) + ".bin")
                if bin_path.exists():
                    bin_path.rename(mapping)
            for suffix in ("mapping", "taxonomy"):
                src = db_dir / f"uniref30_2302_db_{suffix}"
                link = db_dir / f"uniref30_2302_db.idx_{suffix}"
                if src.exists() and not link.exists():
                    link.symlink_to(src.name)
            tax_tar.unlink(missing_ok=True)

        tarball.unlink(missing_ok=True)
        ready.touch()
        click.echo(f"{name}: ready")

    click.echo(f"\nDatabases: {db_dir}")
    click.echo("Predictions will auto-detect this database, or pass explicitly:")
    click.echo(f"  tt-boltz predict input.yaml --msa_db_path {db_dir}")


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
@click.option("--use_msa_server", is_flag=True, help="Generate MSAs via ColabFold API (requires internet)")
@click.option("--msa_db_path", default=None, type=click.Path(exists=True), help="Local ColabFold DB for offline MSA (default: auto-detect ~/.boltz/msa_db)")
@click.option("--use_envdb", is_flag=True, help="Also search ColabFold environmental database (requires envdb)")
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
@click.option("--num_devices", default=0, type=int, help="Number of TT devices to use (0=all available)")
@click.option("--device_ids", default=None, type=str, help="Comma-separated TT device IDs to use (e.g. '0,2')")
@click.option("--debug", is_flag=True, help="Debug mode: no Rich display, no output suppression")
@click.option("--log", is_flag=True, help="With --debug: print per-device stage progress")
def predict(data, out_dir, cache, checkpoint, accelerator, recycling_steps, sampling_steps,
            diffusion_samples, max_parallel_samples, step_scale, output_format, override,
            seed, use_msa_server, msa_db_path, use_envdb, msa_server_url, msa_pairing_strategy,
            msa_server_username, msa_server_password, api_key_value, use_potentials,
            method, max_msa_seqs, subsample_msa, num_subsampled_msa, no_kernels, trace,
            write_pae, write_pde, write_embeddings, affinity_mw_correction,
            sampling_steps_affinity, diffusion_samples_affinity, affinity_checkpoint,
            num_devices, device_ids, debug, log):
    """Run Boltz-2 structure prediction.

    DATA is a YAML/FASTA file or a directory of them.
    Model stays in memory across all predictions. Resume by re-running (skips existing outputs).

    \b
    Output:
        msa/                # MSA cache (keyed by sequence hash, shared across runs)
        boltz_results_<name>/
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

    # Auto-detect local MSA DB (priority: --msa_db_path > ~/.boltz/msa_db)
    if not msa_db_path and not use_msa_server:
        default_msa_db = cache / "msa_db"
        if (default_msa_db / "UNIREF30_READY").exists():
            msa_db_path = str(default_msa_db)

    if use_envdb and use_msa_server:
        click.echo("Note: --use_envdb is only used with offline MSA; ignored with --use_msa_server")

    if use_envdb and not use_msa_server and not msa_db_path:
        raise RuntimeError(
            "--use_envdb requires offline MSA DB setup.\n"
            "Run: tt-boltz msa --db all  (or use --use_msa_server)"
        )

    if msa_db_path and not use_msa_server:
        _validate_offline_msa_db(Path(msa_db_path), require_envdb=use_envdb)

    if use_msa_server:
        msa_server_username = msa_server_username or os.environ.get("BOLTZ_MSA_USERNAME")
        msa_server_password = msa_server_password or os.environ.get("BOLTZ_MSA_PASSWORD")
        api_key_value = api_key_value or os.environ.get("MSA_API_KEY_VALUE")

    data = Path(data).expanduser()
    out = Path(out_dir).expanduser() / f"boltz_results_{data.stem}"
    msa_dir = Path(out_dir).expanduser() / "msa"
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

    torch_device = torch.device("cuda:0" if accelerator == "gpu" and torch.cuda.is_available() else "cpu")

    # Detect TT devices via filesystem (no ttnn import — avoids PCIe lock in parent)
    if use_tt:
        import glob as _glob
        all_devices = sorted(int(p.rsplit("/", 1)[-1]) for p in _glob.glob("/dev/tenstorrent/[0-9]*"))
        if device_ids:
            devices = [int(d.strip()) for d in device_ids.split(",")]
        elif num_devices > 0:
            devices = all_devices[:num_devices]
        else:
            devices = all_devices
        devices = devices[:len(files)]
        n_devices = max(1, len(devices))
    else:
        devices = [0]
        n_devices = 1


    click.echo("")
    click.echo("")
    click.echo("")

    # --- Model kwargs (built once, shared by single- and multi-device paths) ---
    conf_ckpt = str(checkpoint or cache / "boltz2_conf.ckpt")
    aff_ckpt = str(affinity_checkpoint or cache / "boltz2_aff.ckpt")

    _diffusion = {"step_scale": step_scale or 1.5, "gamma_0": 0.8, "gamma_min": 1.0,
                  "noise_scale": 1.003, "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0,
                  "sigma_data": 16.0, "P_mean": -1.2, "P_std": 1.5,
                  "coordinate_augmentation": True, "alignment_reverse_diff": True,
                  "synchronize_sigmas": True}
    _pairformer = {"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True}
    _msa = {"subsample_msa": subsample_msa, "num_subsampled_msa": num_subsampled_msa,
            "use_paired_feature": True}

    conf_kwargs = dict(
        predict_args={"recycling_steps": recycling_steps, "sampling_steps": sampling_steps,
                      "diffusion_samples": diffusion_samples, "max_parallel_samples": max_parallel_samples},
        diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
        steering_args={"fk_steering": use_potentials, "physical_guidance_update": use_potentials,
                       "contact_guidance_update": True, "num_particles": 3, "fk_lambda": 4.0,
                       "fk_resampling_interval": 3, "num_gd_steps": 20},
        use_kernels=not no_kernels, use_tenstorrent=use_tt, trace=trace,
    )
    aff_kwargs = dict(
        predict_args={"recycling_steps": 5, "sampling_steps": sampling_steps_affinity,
                      "diffusion_samples": diffusion_samples_affinity, "max_parallel_samples": 1},
        diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
        steering_args={"fk_steering": False, "physical_guidance_update": False,
                       "contact_guidance_update": False, "num_particles": 3, "fk_lambda": 4.0,
                       "fk_resampling_interval": 3, "num_gd_steps": 20},
        affinity_mw_correction=affinity_mw_correction, use_tenstorrent=use_tt, trace=trace,
    )

    results_path = out / "results.json"
    results = [] if override or not results_path.exists() else json.loads(results_path.read_text())

    # =====================================================================
    # TT path — reuses _predict_worker for all cases:
    #   1 device  → call worker directly (no subprocess overhead)
    #   N devices → one subprocess per card, shared kernel cache
    # =====================================================================
    if use_tt:
        worker_cfg = {
            "conf_ckpt": conf_ckpt, "aff_ckpt": aff_ckpt,
            "conf_kwargs": conf_kwargs, "aff_kwargs": aff_kwargs,
            "mol_dir": str(mol_dir), "msa_dir": str(msa_dir), "struct_dir": str(struct_dir),
            "method": method, "output_format": output_format,
            "write_pae": write_pae, "write_pde": write_pde, "write_embeddings": write_embeddings,
            "use_msa_server": use_msa_server, "msa_db_path": msa_db_path, "use_envdb": use_envdb,
            "msa_server_url": msa_server_url, "msa_pairing_strategy": msa_pairing_strategy,
            "msa_server_username": msa_server_username, "msa_server_password": msa_server_password,
            "api_key_value": api_key_value, "max_msa_seqs": max_msa_seqs,
            "results_path": str(results_path),
        }
        import sys as _sys
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        failed = 0

        pq = ctx.Queue()
        suppress = not debug
        display = (ProgressDisplay(pq, total=len(files), n_devices=n_devices) if not debug
                   else DebugDisplay(pq) if log else NullDisplay(pq))
        display.start()

        try:
            if n_devices == 1:
                os.environ["TT_VISIBLE_DEVICES"] = str(devices[0])
                _predict_worker(devices[0], [str(x) for x in files], worker_cfg, q, pq,
                                suppress_output=suppress)
                msg = q.get()
                results.extend(msg.get("results", []))
                failed = msg.get("failed", 0)
            else:
                file_buckets = [files[i::n_devices] for i in range(n_devices)]
                watchdog_q = ctx.Queue()
                worker_cfg["watchdog_queue"] = watchdog_q

                # Per-device worker state for restart-on-hang.
                states = {}
                procs = {}
                max_retries = 2
                idle_timeout_s = 180.0
                check_log_interval_s = 60.0
                click.echo(f"[watchdog] enabled: reset worker after {int(idle_timeout_s)}s without updates")

                def _spawn(dev: int):
                    st = states[dev]
                    remaining = st["bucket"][st["next_idx"]:]
                    if not remaining:
                        procs.pop(dev, None)
                        return
                    p = ctx.Process(
                        target=_predict_worker,
                        args=(dev, [str(x) for x in remaining], worker_cfg, q, pq),
                        kwargs={"suppress_output": suppress},
                    )
                    p.start()
                    procs[dev] = p
                    st["last_event"] = time.time()
                    st["current"] = None

                for i, bucket in enumerate(file_buckets):
                    if not bucket:
                        continue
                    dev = devices[i]
                    states[dev] = {
                        "bucket": bucket,
                        "next_idx": 0,
                        "current": None,
                        "last_event": time.time(),
                        "last_check_log": 0.0,
                        "retries": {},
                    }
                    _spawn(dev)

                def _advance_done(st, name: str):
                    while st["next_idx"] < len(st["bucket"]) and st["bucket"][st["next_idx"]].stem != name:
                        st["next_idx"] += 1
                    if st["next_idx"] < len(st["bucket"]) and st["bucket"][st["next_idx"]].stem == name:
                        st["next_idx"] += 1

                while procs:
                    # Drain watchdog progress events.
                    while True:
                        try:
                            ev = watchdog_q.get_nowait()
                        except Empty:
                            break
                        dev = ev.get("dev")
                        if dev not in states:
                            continue
                        st = states[dev]
                        st["last_event"] = time.time()
                        if ev.get("event") == "start":
                            st["current"] = ev.get("name")
                        elif ev.get("event") == "done":
                            _advance_done(st, ev.get("name", ""))
                            st["current"] = None

                    # Drain worker completion messages.
                    while True:
                        try:
                            msg = q.get_nowait()
                        except Empty:
                            break
                        dev = msg.get("dev")
                        results.extend(msg.get("results", []))
                        failed += msg.get("failed", 0)
                        if dev in states:
                            # Worker processed all currently assigned files.
                            states[dev]["next_idx"] = len(states[dev]["bucket"])
                            states[dev]["current"] = None
                        p = procs.pop(dev, None)
                        if p is not None:
                            p.join(timeout=1)
                        if dev in states:
                            _spawn(dev)

                    # Watchdog: no update for 3 minutes => restart just that worker/device.
                    now = time.time()
                    for dev, p in list(procs.items()):
                        st = states[dev]
                        if not p.is_alive():
                            continue
                        if st["current"] is None:
                            continue
                        idle_s = now - st["last_event"]
                        if idle_s >= 60 and now - st["last_check_log"] >= check_log_interval_s:
                            click.echo(f"[watchdog] check device {dev}: {st['current']} idle {int(idle_s)}s")
                            st["last_check_log"] = now
                        if idle_s <= idle_timeout_s:
                            continue

                        target = st["current"]
                        tries = st["retries"].get(target, 0) + 1
                        st["retries"][target] = tries
                        click.echo(f"\n[watchdog] device {dev} stalled on {target} (>{int(idle_timeout_s)}s no updates)")

                        p.terminate()
                        p.join(timeout=10)
                        if p.is_alive():
                            p.kill()
                            p.join(timeout=5)
                        # Allow driver handles to drain before running tt-smi reset.
                        time.sleep(1.0)
                        procs.pop(dev, None)

                        if tries > max_retries:
                            click.echo(f"[watchdog] giving up on {target} after {max_retries} retries")
                            row = {"id": target, "status": "failed", "error": "watchdog timeout"}
                            results.append(row)
                            failed += 1
                            if "results_path" in worker_cfg:
                                try:
                                    _append_result(row, Path(worker_cfg["results_path"]))
                                except Exception:
                                    pass
                            _advance_done(st, target)
                            st["current"] = None
                            _spawn(dev)
                            continue

                        click.echo(f"[watchdog] resetting device {dev} and retrying {target} (attempt {tries}/{max_retries})")
                        try:
                            _reset_tt_device(dev)
                            time.sleep(2.0)
                        except Exception as e:
                            click.echo(f"[watchdog] device reset failed on {dev}: {e}")
                        st["last_event"] = time.time()
                        st["current"] = None
                        _spawn(dev)

                    time.sleep(0.5)
        finally:
            _sys.stdout = _sys.__stdout__
            display.stop()

    # =====================================================================
    # CPU / GPU path: inline, single process (with progress display)
    # =====================================================================
    else:
        import sys as _sys

        tokenizer, featurizer = Boltz2Tokenizer(), Boltz2Featurizer()
        ccd = load_canonicals(mol_dir)
        prepare = partial(prepare_features, ccd=ccd, mol_dir=mol_dir, msa_dir=msa_dir,
                          tokenizer=tokenizer, featurizer=featurizer,
                          use_msa=use_msa_server, msa_url=msa_server_url,
                          msa_strategy=msa_pairing_strategy, msa_user=msa_server_username,
                          msa_pass=msa_server_password, api_key=api_key_value,
                          max_msa=max_msa_seqs, msa_db_path=msa_db_path,
                          use_envdb=use_envdb)

        from queue import Queue as ThreadQueue
        pq = ThreadQueue()
        display = (ProgressDisplay(pq, total=len(files), n_devices=1) if not debug
                   else DebugDisplay(pq) if log else NullDisplay(pq))
        display.start()
        pq.put({"dev": 0, "event": "init", "assigned": len(files)})

        _saved_stdout = _sys.stdout
        if not debug:
            _sys.stdout = open(os.devnull, "w")

        try:
            pq.put({"dev": 0, "event": "loading"})
            model = Boltz2.load_from_checkpoint(conf_ckpt, **conf_kwargs).eval().to(torch_device)
            model.progress_fn = make_progress_fn(pq, 0)

            affinity_queue = []
            failed = 0
            for path in files:
                row = {"id": path.stem, "status": "failed"}
                t0 = time.time()
                pq.put({"dev": 0, "event": "start", "name": path.stem})
                try:
                    pq.put({"dev": 0, "event": "stage", "stage": "msa"})
                    feats, input_struct = prepare(path, method=method)
                    batch = to_batch(feats, torch_device)
                    with torch.no_grad(), torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                        pred = model.predict_step(batch)
                    pq.put({"dev": 0, "event": "stage", "stage": "saving"})
                    metrics, best = write_result(pred, batch, input_struct, struct_dir,
                                                 output_format, write_pae, write_pde, write_embeddings)
                    if metrics:
                        row.update(metrics)
                        row["status"] = "ok"
                        row["runtime_s"] = round(time.time() - t0, 1)
                        if feats["record"].affinity and best is not None:
                            affinity_queue.append((path, best))
                except Exception as e:
                    traceback.print_exc()
                    row["error"] = str(e)[:80]
                elapsed = round(time.time() - t0, 1)
                if row["status"] != "ok":
                    failed += 1
                pq.put({"dev": 0, "event": "done", "name": path.stem,
                        "time": elapsed, "status": row["status"],
                        "error": row.get("error", "")})
                results.append(row)
                _save_results(results, results_path)
            del model

            if affinity_queue:
                aff_model = Boltz2.load_from_checkpoint(aff_ckpt, **aff_kwargs).eval().to(torch_device)
                rows_by_id = {r["id"]: r for r in results}
                aff_keys = ["affinity_pred_value", "affinity_probability_binary",
                            "affinity_pred_value1", "affinity_probability_binary1",
                            "affinity_pred_value2", "affinity_probability_binary2"]
                for path, pred_struct in affinity_queue:
                    try:
                        feats, _ = prepare(path, method="other", affinity=True, pred_structure=pred_struct)
                        batch = to_batch(feats, torch_device)
                        with torch.no_grad(), torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                            pred = aff_model.predict_step(batch)
                        if not pred["exception"] and path.stem in rows_by_id:
                            for ak in aff_keys:
                                if ak in pred:
                                    rows_by_id[path.stem][ak] = round(pred[ak].item(), 6)
                    except Exception as e:
                        traceback.print_exc()
        finally:
            _sys.stdout = _saved_stdout
            display.stop()

    # Preserve per-target rows that may have been atomically appended by workers.
    existing_rows = json.loads(results_path.read_text()) if results_path.exists() else []
    merged = {r["id"]: r for r in existing_rows}
    merged.update({r["id"]: r for r in results})
    _save_results(list(merged.values()), results_path)
    click.echo(f"\nDone: {len(files) - failed} ok, {failed} failed — {results_path}")


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
