"""Boltz-2 structure prediction CLI."""

# Suppress noisy ttnn/loguru output before any import pulls in ttnn.
# These are defaults — --debug mode removes them so everything is visible.
import os as _os, sys as _sys
if "--debug" not in _sys.argv:
    _os.environ.setdefault("LOGURU_LEVEL", "WARNING")
    _os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")


def _install_nanobind_leak_stderr_filter() -> None:
    """Drop nanobind leak reports while forwarding other fd-level stderr."""
    try:
        read_fd, write_fd = _os.pipe()
        original_stderr_fd = _os.dup(2)
        pid = _os.fork()
        if pid == 0:
            try:
                _os.close(write_fd)
                suppressing_nanobind_leak = False
                with _os.fdopen(read_fd, "rb", closefd=True) as pipe:
                    for raw_line in pipe:
                        line = raw_line.decode("utf-8", errors="replace")
                        if line.startswith("nanobind: leaked "):
                            suppressing_nanobind_leak = True
                            continue
                        if suppressing_nanobind_leak:
                            if (
                                line.startswith(" - ")
                                or line.startswith("nanobind: this is likely caused")
                                or line.startswith("See https://nanobind.")
                            ):
                                continue
                            suppressing_nanobind_leak = False
                        _os.write(original_stderr_fd, raw_line)
            except Exception:
                pass
            finally:
                _os._exit(0)

        _os.close(read_fd)
        _os.dup2(write_fd, 2)
        _os.close(write_fd)
        python_stderr = _os.fdopen(
            _os.dup(original_stderr_fd),
            "w",
            buffering=1,
            encoding=getattr(_sys.stderr, "encoding", None) or "utf-8",
            errors=getattr(_sys.stderr, "errors", None) or "replace",
        )
        _sys.stderr = python_stderr
        _sys.__stderr__ = python_stderr
        _os.close(original_stderr_fd)
    except Exception:
        pass


if "--debug" not in _sys.argv:
    _install_nanobind_leak_stderr_filter()


import base64
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import random
import signal
import shutil
import tempfile
import subprocess
import tarfile
import time
import urllib.request
import warnings
import fcntl
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import click
import numpy as np
import torch
from rdkit import Chem

from tt_boltz.data import const
from tt_boltz.data.featurizer import Boltz2Featurizer
from tt_boltz.data.mol import load_canonicals, load_molecules
from tt_boltz.data.msa import run_mmseqs2
from tt_boltz.data.parse import parse_a3m, parse_csv, parse_fasta, parse_yaml
from tt_boltz.data.tokenize import Boltz2Tokenizer
from tt_boltz.data.types import Coords, Input, Interface
from tt_boltz.data.write import to_mmcif, to_pdb
from tt_boltz.boltz2 import Boltz2
from tt_boltz.distributed import (
    ControllerClient,
    ControllerServer,
    job_payloads,
    worker_payload,
)
from tt_boltz.energy import DEFAULT_ENERGY_SAMPLE_HZ, PowerProfiler
from tt_boltz.progress import DebugDisplay, NullDisplay, ProgressDisplay
from tt_boltz.runtime import (
    build_local_workers,
    detect_tenstorrent_devices,
    discover_jobs,
)
from tt_boltz.worker import run_worker_loop

ARTIFACT_BASE_URL = "https://storage.googleapis.com/tt-boltz-artifacts"
URLS = {
    "mols": f"{ARTIFACT_BASE_URL}/mols.tar",
    "conf": f"{ARTIFACT_BASE_URL}/boltz2_conf.ckpt",
    "aff": f"{ARTIFACT_BASE_URL}/boltz2_aff.ckpt",
}


def download(url: str, dest: Path) -> None:
    """Download a required artifact if it is missing locally."""
    if dest.exists():
        return
    click.echo(f"Downloading {dest.name}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        raise RuntimeError(f"Failed to download {dest.name} from {url}") from e


def download_all(cache: Path) -> None:
    """Download all required model files and molecules."""
    tar_path = cache / "mols.tar"
    if not tar_path.exists():
        click.echo(f"Downloading {tar_path.name}")
        urllib.request.urlretrieve(URLS["mols"], tar_path)
    if not (cache / "mols").exists():
        click.echo(f"Extracting {tar_path.name}")
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


def _ensure_pigz(pixi: str) -> None:
    """Install pigz via pixi if not already available."""
    if shutil.which("pigz"):
        return
    click.echo("Installing pigz for fast parallel extraction ...")
    subprocess.run([pixi, "global", "install", "pigz"], check=True)


def _ensure_offline_tools(install_tools: bool) -> None:
    """Ensure mmseqs + colabfold_search + aria2/pigz are available; optionally install them."""
    missing = _missing_offline_tools()
    need_aria2 = not shutil.which("aria2c")
    need_pigz = not shutil.which("pigz")

    if not missing and not need_aria2 and not need_pigz:
        return
    if not install_tools:
        all_missing = missing + (["aria2c"] if need_aria2 else []) + (["pigz"] if need_pigz else [])
        raise RuntimeError(
            "Missing offline MSA tools: " + ", ".join(all_missing) + "\n"
            "Rerun with: tt-boltz msa --install-tools"
        )

    pixi = _ensure_pixi()
    _ensure_aria2(pixi)
    _ensure_pigz(pixi)

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


def _find_mmseqs() -> str | None:
    """Find mmseqs binary on PATH or at common install locations."""
    found = shutil.which("mmseqs")
    if found:
        return found
    for p in _MMSEQS_SEARCH_PATHS:
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


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


def _recommended_threads() -> int:
    """Pick a conservative-but-fast thread count across machine sizes."""
    return max(1, int(os.cpu_count() or 1))


def _extract_tarball(tarball: Path, out_dir: Path) -> None:
    """Extract tar.gz, using pigz for parallel decompression if available."""
    threads = _recommended_threads()
    pigz = shutil.which("pigz")
    if pigz:
        # GNU tar supports --use-compress-program/-I with an argument string.
        cmd = ["tar", "-I", f"{pigz} -d -p {threads}", "-xf", str(tarball), "-C", str(out_dir)]
    else:
        cmd = ["tar", "-xzf", str(tarball), "-C", str(out_dir)]
    subprocess.run(cmd, check=True)


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
    mmseqs_bin = _find_mmseqs()
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
        cmd_base = [
            colabfold_bin, str(fasta), db_path, str(a3m_out),
            "--use-env", "1" if use_env else "0", "--use-templates", "0",
            "--db-load-mode", "2", "--threads", str(os.cpu_count() or 1),
        ]
        if len(seqs) > 1:
            cmd_base += ["--pair-mode", "unpaired_paired", "--pairing_strategy", strategy_val]

        commands = []
        if mmseqs_bin:
            commands.append(cmd_base[:4] + ["--mmseqs", mmseqs_bin] + cmd_base[4:])
        commands.append(cmd_base)

        last_error = ""
        for idx, cmd in enumerate(commands):
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                last_error = ""
                break
            err = (result.stderr or result.stdout or "").strip()
            last_error = "\n".join(err.splitlines()[-20:]) if err else ""
            if idx < len(commands) - 1:
                click.echo("  colabfold_search failed with explicit --mmseqs, retrying with default lookup")
        if last_error:
            raise RuntimeError(
                f"colabfold_search failed (exit {result.returncode})\n{last_error}"
            )
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
                     affinity=False, pred_structure=None, rng_seed=42):
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
        tok, np.random.default_rng(rng_seed), mols, False, const.max_msa_seqs,
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
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(content)
        with open(tmp, "r+") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


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


def _results_lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def _results_backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".bak")


def _load_results_resilient(path: Path) -> list[dict]:
    """Load results.json safely; fall back to .bak if corrupted."""
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        # Keep a copy of the corrupted file for post-mortem debugging.
        ts = time.strftime("%Y%m%d-%H%M%S")
        corrupt_copy = path.with_suffix(path.suffix + f".corrupt-{ts}")
        try:
            shutil.copy2(path, corrupt_copy)
        except Exception:
            pass

        bak = _results_backup_path(path)
        if bak.exists():
            try:
                data = json.loads(bak.read_text())
                return data if isinstance(data, list) else []
            except Exception:
                pass
        return []


def _save_results_unlocked(results: list[dict], path: Path) -> None:
    """Write results.json with backup; caller must hold lock."""
    bak = _results_backup_path(path)
    if path.exists():
        try:
            shutil.copy2(path, bak)
        except Exception:
            pass
    _atomic_write(path, json.dumps(results, indent=2))


def _save_results(results: list[dict], path: Path) -> None:
    """Save results with inter-process lock, backup, and atomic replace."""
    lock_path = _results_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            _save_results_unlocked(results, path)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _append_result(row: dict, path: Path) -> None:
    """Append one result row to results.json atomically.

    Safe for concurrent workers: reads existing, merges, writes via rename.
    """
    lock_path = _results_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            existing = _load_results_resilient(path)
            existing = [r for r in existing if isinstance(r, dict) and r.get("id") != row["id"]]
            existing.append(row)
            _save_results_unlocked(existing, path)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _detect_p300_devices() -> list[int]:
    """Return P300 TT device indices from kernel sysfs.

    This avoids importing ttnn in the parent process and avoids requiring tt-smi.
    The subsystem IDs mirror tt-metal's Blackhole board-type mapping for P300.
    """
    p300_subsystems = {"0x0044", "0x0045", "0x0046"}
    devices = []
    for entry in (Path("/sys/class/tenstorrent")).glob("tenstorrent!*"):
        try:
            device_id = int(entry.name.rsplit("!", 1)[1])
            subsystem_id = (entry / "device" / "subsystem_device").read_text().strip().lower()
        except Exception:
            continue
        if subsystem_id in p300_subsystems:
            devices.append(device_id)
    return sorted(devices)


def _find_ttnn_mesh_graph_descriptor(filename: str) -> str | None:
    spec = importlib.util.find_spec("ttnn")
    if spec is None or not spec.submodule_search_locations:
        return None
    ttnn_root = Path(next(iter(spec.submodule_search_locations)))
    descriptor = ttnn_root / "tt_metal" / "fabric" / "mesh_graph_descriptors" / filename
    return str(descriptor) if descriptor.is_file() else None


def _build_worker_device_assignments(devices: list[int]) -> dict[int, dict[str, object]]:
    """Build per-worker visibility/logical-device assignments.

    P300 chips are exposed one-at-a-time like other devices. Because a lone P300
    chip is a custom topology, those workers also get a 1x1 Blackhole MGD.
    """
    p300_devices = set(_detect_p300_devices())
    p300_mgd = (
        _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if p300_devices and not os.environ.get("TT_MESH_GRAPH_DESC_PATH")
        else None
    )

    assignments: dict[int, dict[str, object]] = {}
    for device in devices:
        assignment: dict[str, object] = {"visible_devices": str(device), "logical_device_id": 0}
        if device in p300_devices and p300_mgd:
            assignment["mesh_graph_descriptor"] = p300_mgd
        assignments[device] = assignment
    return assignments


def _local_workers(accelerator: str, num_devices: int, device_ids: str | None, max_workers: int) -> list:
    """Build a list of WorkerSlot objects covering this host's accelerators."""
    if accelerator != "tenstorrent":
        return build_local_workers(accelerator, [object()], [0])
    devices = detect_tenstorrent_devices(device_ids, num_devices, max_workers=max_workers)
    if not devices:
        raise RuntimeError(
            "No Tenstorrent devices found. Use --accelerator cpu/gpu or check /dev/tenstorrent."
        )
    workers = build_local_workers("tenstorrent", [object()] * len(devices), devices)
    assigns = _build_worker_device_assignments([int(w.device_id) for w in workers])
    return [
        replace(
            w,
            visible_devices=str(assigns[int(w.device_id)]["visible_devices"]),
            logical_device_id=int(assigns[int(w.device_id)]["logical_device_id"]),
            mesh_graph_descriptor=assigns[int(w.device_id)].get("mesh_graph_descriptor"),
        )
        for w in workers
    ]


def _spawn_worker_processes(controller_url: str, workers: list, debug: bool) -> list:
    """Spawn one process per worker slot, each connected to the controller."""
    ctx = mp.get_context("spawn")
    procs = []
    for worker in workers:
        proc = ctx.Process(
            target=run_worker_loop,
            args=(controller_url, worker_payload(worker), debug),
        )
        proc.start()
        procs.append(proc)
    return procs


def _stop_worker_processes(procs: list) -> None:
    """Cleanly stop spawned worker processes."""
    for proc in procs:
        if proc.is_alive():
            try:
                os.kill(proc.pid, signal.SIGINT)
            except Exception:
                pass
            proc.join(timeout=12)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=8)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=3)


def _parse_listen(listen: str | None) -> tuple[str, int]:
    """Parse a --listen value into (host, port). Defaults are 0.0.0.0:8765."""
    if not listen:
        return "127.0.0.1", 0
    listen = listen.strip()
    if listen.isdigit():
        return "0.0.0.0", int(listen)
    if ":" in listen:
        host, _, port = listen.rpartition(":")
        return (host or "0.0.0.0"), int(port)
    return listen, 8765


def _public_join_url(bind_host: str, port: int) -> str:
    """Best-effort host name to print so remote workers can connect."""
    if bind_host not in ("0.0.0.0", "::", ""):
        return f"http://{bind_host}:{port}"
    try:
        import socket

        return f"http://{socket.gethostname()}:{port}"
    except Exception:
        return f"http://<this-host>:{port}"


def _stream_run(client: ControllerClient, run_id: str, total: int, n_workers: int,
                debug: bool, log: bool, results_path: Path | None = None,
                struct_dir: Path | None = None) -> int:
    """Stream events from a controller and render progress; return failed count.

    On every done event we fetch that job's output files from the controller
    and write them under struct_dir, and merge its result row into
    results.json. Interrupted runs preserve every protein finished so far.
    """
    from queue import Queue as ThreadQueue

    pq = ThreadQueue()
    display = (
        ProgressDisplay(pq, total=total, n_workers=n_workers) if not debug
        else DebugDisplay(pq) if log else NullDisplay(pq)
    )
    display.start()
    after = 0
    failed = 0
    rows_by_id: dict[str, dict] = {}
    if results_path is not None:
        rows_by_id = {r["id"]: r for r in _load_results_resilient(results_path)
                      if isinstance(r, dict) and "id" in r}
    if struct_dir is not None:
        struct_dir.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            snapshot = client.events(run_id, after)
            for ev in snapshot.get("events", []):
                after = max(after, int(ev.get("seq", after)))
                if ev.get("event") in ("run", "run_done"):
                    continue
                if ev.get("event") == "done":
                    row = ev.get("row")
                    if isinstance(row, dict) and "id" in row:
                        if results_path is not None:
                            rows_by_id[row["id"]] = row
                            try:
                                _save_results(list(rows_by_id.values()), results_path)
                            except Exception:
                                pass
                        if struct_dir is not None and row.get("status") == "ok":
                            _write_job_outputs(client, run_id, row["id"], struct_dir)
                pq.put(ev)
            if snapshot.get("status") in ("ok", "failed"):
                failed = int(snapshot.get("failed") or 0)
                break
            time.sleep(0.5)
    finally:
        display.stop()
    return failed


def _write_job_outputs(client: ControllerClient, run_id: str, job_id: str,
                       struct_dir: Path) -> None:
    """Fetch a completed job's output files and write them under struct_dir."""
    try:
        outputs = client.job_outputs(run_id, job_id) or {}
    except Exception:
        return
    for name, content_b64 in outputs.items():
        if not content_b64:
            continue
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            continue
        target = struct_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_bytes(base64.b64decode(content_b64))
        except Exception:
            pass


@contextmanager
def _scheduler_session(listen: str | None, workers: list, debug: bool):
    """Start an in-process scheduler, spawn local worker subprocesses against
    it, and yield (client, public_join_url) for the duration of the run.

    The scheduler keeps its SQLite state in a private temp directory and
    discards it on exit, so a run never leaves bookkeeping artifacts in the
    user's results directory. public_join_url is None unless --listen was
    passed; when set, it's the address a remote `tt-boltz worker --connect
    ...` should target.
    """
    listen_host, listen_port = _parse_listen(listen)
    tmpdir = Path(tempfile.mkdtemp(prefix="tt-boltz-scheduler-"))
    db_path = tmpdir / "controller.sqlite3"
    server = ControllerServer(listen_host, listen_port, db_path)
    server.serve_in_background()
    url = f"http://127.0.0.1:{server.port}"
    public_url = _public_join_url(listen_host, server.port) if listen else None
    procs = _spawn_worker_processes(url, workers, debug)
    try:
        yield ControllerClient(url), public_url
    finally:
        _stop_worker_processes(procs)
        server.shutdown()
        shutil.rmtree(tmpdir, ignore_errors=True)


def _persist_run_results(client: ControllerClient, run_id: str, results_path: Path) -> None:
    """Merge per-run result rows from the controller into the local results.json."""
    try:
        new_rows = client.results(run_id)
    except Exception:
        return
    existing = _load_results_resilient(results_path)
    merged = {r["id"]: r for r in existing if isinstance(r, dict) and "id" in r}
    for row in new_rows:
        if isinstance(row, dict) and "id" in row:
            merged[row["id"]] = row
    _save_results(list(merged.values()), results_path)



@click.group()
def cli():
    """Run Boltz-2 inference on Tenstorrent hardware."""


@cli.command("install-deps")
def install_deps():
    """Install system dependencies that match the installed ttnn wheel."""
    from tt_boltz.install_system_deps import main as install_system_deps

    install_system_deps()


@cli.command("worker")
@click.option("--connect", required=True, help="Controller URL, e.g. http://HOST:8765")
@click.option("--accelerator", type=click.Choice(["gpu", "cpu", "tenstorrent"]), default="tenstorrent")
@click.option("--num_devices", default=0, type=int, help="Number of TT devices to use (0=all available)")
@click.option("--device_ids", default=None, type=str, help="Comma-separated TT device IDs to use")
@click.option("--debug", is_flag=True, help="Do not suppress worker output")
def worker_cmd(connect, accelerator, num_devices, device_ids, debug):
    """Join a tt-boltz controller and run predictions on this machine's accelerators."""
    workers = _local_workers(accelerator, num_devices, device_ids, max_workers=10_000)
    click.echo(f"Connecting {len(workers)} worker{'s' if len(workers) != 1 else ''} to {connect}")
    if accelerator == "tenstorrent":
        click.echo(f"  Devices: {[int(w.device_id) for w in workers]}")
    for worker in workers:
        click.echo(f"  {worker.label}")

    procs = _spawn_worker_processes(connect, workers, debug)
    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        click.echo("\nStopping workers...")
        _stop_worker_processes(procs)


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
        if tarball.exists() and tarball.stat().st_size > 0:
            click.echo(f"  Reusing existing tarball: {tarball.name}")
        else:
            _download_file(info["url"], tarball)

        click.echo(f"{name}: extracting")
        _extract_tarball(tarball, db_dir)

        if _mmseqs_index_exists(db_dir, info["name"]):
            click.echo(f"{name}: index already present")
        else:
            threads = _recommended_threads()
            click.echo(f"{name}: building index (this takes a while)")
            subprocess.run(
                [mmseqs, "createindex", str(db_dir / info["name"]),
                 str(db_dir / f"tmp_{name}"), "--remove-tmp-files", "1",
                 "--threads", str(threads)],
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
@click.option("--trace", is_flag=True, help="Print Boltz2 module-stage debug messages")
@click.option("--no-tracing", "no_tracing", is_flag=True, help="Disable Pairformer TT-NN trace capture+replay")
@click.option("--write_pae", is_flag=True, help="Write PAE matrix per target")
@click.option("--write_pde", is_flag=True, help="Write PDE matrix per target")
@click.option("--write_embeddings", is_flag=True, help="Write s/z embeddings per target")
@click.option("--affinity_mw_correction", is_flag=True)
@click.option("--sampling_steps_affinity", default=200, type=int)
@click.option("--diffusion_samples_affinity", default=5, type=int)
@click.option("--affinity_checkpoint", type=click.Path(exists=True), default=None)
@click.option("--num_devices", default=0, type=int, help="Number of TT devices to use (0=all available)")
@click.option("--device_ids", default=None, type=str, help="Comma-separated TT device IDs to use (e.g. '0,2')")
@click.option("--fast", is_flag=True, help="Use block-fp8 for some operations (slightly lower precision, faster)")
@click.option("--debug", is_flag=True, help="Debug mode: no Rich display, no output suppression")
@click.option("--log", is_flag=True, help="With --debug: print per-device stage progress")
@click.option("--report-energy", "report_energy", is_flag=True, help="Report TT device energy and write a power-vs-time plot (single-device TT runs)")
@click.option("--energy-sample-hz", "energy_sample_hz", default=DEFAULT_ENERGY_SAMPLE_HZ, type=float, show_default=True, help="Sampling rate in Hz for power reporting")
@click.option("--energy-metric", "energy_metric", default="both", type=click.Choice(["both", "tdp", "input"]), show_default=True, help="Which power channel(s) to measure")
@click.option("--listen", default=None, help="Bind scheduler to HOST:PORT so remote workers can join (e.g. 8765 or 0.0.0.0:8765)")
def predict(data, out_dir, cache, checkpoint, accelerator, recycling_steps, sampling_steps,
            diffusion_samples, max_parallel_samples, step_scale, output_format, override,
            seed, use_msa_server, msa_db_path, use_envdb, msa_server_url, msa_pairing_strategy,
            msa_server_username, msa_server_password, api_key_value, use_potentials,
            method, max_msa_seqs, subsample_msa, num_subsampled_msa, no_kernels, trace,
            no_tracing,
            write_pae, write_pde, write_embeddings, affinity_mw_correction,
            sampling_steps_affinity, diffusion_samples_affinity, affinity_checkpoint,
            num_devices, device_ids, fast, debug, log,
            report_energy, energy_sample_hz, energy_metric, listen):
    """Run Boltz-2 structure prediction.

    DATA is a YAML/FASTA file or a directory of them.
    A scheduler runs in-process and dispatches jobs to local workers; pass
    --listen to also accept workers from other machines.

    \b
    Output:
        msa/                # MSA cache (keyed by sequence hash)
        boltz_results_<name>/
            structures/     # one CIF per complex (pLDDT in B-factors)
            results.json    # confidence metrics + affinity
    """
    use_tt = accelerator == "tenstorrent"
    if fast and not use_tt:
        click.echo("Note: --fast is only used with --accelerator tenstorrent; ignoring.")
    warnings.filterwarnings("ignore", ".*Tensor Cores.*")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    os.environ.setdefault("CUEQ_DEFAULT_CONFIG", "1")
    os.environ.setdefault("CUEQ_DISABLE_AOT_TUNING", "1")

    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_all(cache)

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
    out_dir_path = Path(out_dir).expanduser()
    out = out_dir_path / f"boltz_results_{data.stem}"
    msa_dir = out_dir_path / "msa"
    struct_dir = out / "structures"
    msa_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)
    mol_dir = cache / "mols"

    jobs = discover_jobs(data, struct_dir, output_format, override)
    if not jobs:
        all_jobs = discover_jobs(data, struct_dir, output_format, override=True)
        click.echo("All predictions complete" if all_jobs else "No input files found")
        return

    if method and method.lower() not in const.method_types_ids:
        raise ValueError(f"Unknown method: {method}")

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

    worker_cfg = {
        "conf_ckpt": str(checkpoint or cache / "boltz2_conf.ckpt"),
        "aff_ckpt": str(affinity_checkpoint or cache / "boltz2_aff.ckpt"),
        "conf_kwargs": conf_kwargs, "aff_kwargs": aff_kwargs,
        "mol_dir": str(mol_dir), "msa_dir": str(msa_dir), "struct_dir": str(struct_dir),
        "method": method, "output_format": output_format,
        "write_pae": write_pae, "write_pde": write_pde, "write_embeddings": write_embeddings,
        "use_msa_server": use_msa_server, "msa_db_path": msa_db_path, "use_envdb": use_envdb,
        "msa_server_url": msa_server_url, "msa_pairing_strategy": msa_pairing_strategy,
        "msa_server_username": msa_server_username, "msa_server_password": msa_server_password,
        "api_key_value": api_key_value, "max_msa_seqs": max_msa_seqs,
        "seed": seed,
        "fast": fast,
        "no_tracing": no_tracing,
    }
    run_payload = {
        "data": str(data),
        "out_dir": str(out_dir_path),
        "result_dir": str(out),
        "jobs": job_payloads(jobs),
        "config": worker_cfg,
    }

    workers = _local_workers(
        "tenstorrent" if use_tt else accelerator,
        num_devices, device_ids, max_workers=max(len(jobs), 1),
    )
    devices = [int(w.device_id) for w in workers if w.accelerator == "tenstorrent"]

    energy_profiler = None
    if report_energy:
        if not use_tt:
            click.echo("Energy profiling currently requires --accelerator=tenstorrent; skipping")
        elif len(devices) != 1:
            click.echo("Energy profiling currently supports one TT device only; skipping")
        else:
            try:
                energy_profiler = PowerProfiler(
                    device_id=devices[0],
                    sample_hz=energy_sample_hz,
                    input_sample_hz=energy_sample_hz,
                    metric_mode=energy_metric,
                )
                energy_profiler.start()
                click.echo(
                    f"Energy profiler: device={devices[0]} metric={energy_metric} hz={energy_sample_hz:.2f}"
                )
            except Exception as e:
                click.echo(f"Energy profiler unavailable: {e}")
                energy_profiler = None

    with _scheduler_session(listen, workers, debug) as (client, public_url):
        if public_url:
            click.echo(f"Workers may join: tt-boltz worker --connect {public_url}")
        run_id = client.create_run(run_payload)["run_id"]
        failed = _stream_run(client, run_id, total=len(jobs), n_workers=len(workers),
                             debug=debug, log=log, results_path=results_path,
                             struct_dir=struct_dir)
        _persist_run_results(client, run_id, results_path)

    if energy_profiler is not None:
        energy_profiler.stop()
        summary = energy_profiler.summarize()
        energy_csv_path = out / "power_profile.csv"
        energy_plot_path = out / "power_profile.png"
        energy_profiler.write_csv(energy_csv_path)
        click.echo("\nEnergy summary (one TT device)")
        click.echo(f"  device_id:      {devices[0]}")
        if summary.energy_j is not None:
            click.echo("  tdp_metric:")
            click.echo(f"    samples:      {summary.samples}")
            click.echo(f"    duration_s:   {summary.duration_s:.3f}")
            click.echo(f"    energy_j:     {summary.energy_j:.3f}")
            click.echo(f"    energy_wh:    {summary.energy_wh:.6f}")
            click.echo(f"    avg_power_w:  {summary.avg_w:.3f}")
            click.echo(f"    peak_power_w: {summary.peak_w:.3f}")
            click.echo(f"    min_power_w:  {summary.min_w:.3f}")
            click.echo(f"    source:       {energy_profiler.source}")
        if summary.input_energy_j is not None:
            click.echo("  input_power_metric:")
            click.echo(f"    samples:      {summary.input_samples}")
            click.echo(f"    duration_s:   {summary.input_duration_s:.3f}")
            click.echo(f"    energy_j:     {summary.input_energy_j:.3f}")
            click.echo(f"    energy_wh:    {summary.input_energy_wh:.6f}")
            click.echo(f"    avg_power_w:  {summary.input_avg_w:.3f}")
            click.echo(f"    peak_power_w: {summary.input_peak_w:.3f}")
            click.echo(f"    min_power_w:  {summary.input_min_w:.3f}")
            click.echo(f"    source:       {energy_profiler.input_power_source}")
        if energy_profiler.input_power_note:
            click.echo(f"  input_power:    {energy_profiler.input_power_note}")
        click.echo(f"  power_csv:      {energy_csv_path}")
        if energy_profiler.error:
            click.echo(f"  sampler_note:   {energy_profiler.error}")
        wrote_plot = energy_profiler.write_plot(
            energy_plot_path,
            title=f"TT device {devices[0]} power vs time",
        )
        click.echo(f"  power_plot:     {energy_plot_path}" if wrote_plot else "  power_plot:     failed (matplotlib not available)")

    click.echo(f"\nDone: {len(jobs) - failed} ok, {failed} failed — {results_path}")


@cli.command()
@click.option("--max_seq", default=1024, type=int, help="Maximum sequence length to warm up")
@click.option("--max_msa", default=16384, type=int, help="Maximum MSA depth to warm up")
@click.option("--n_samples", default=1, type=int, help="Diffusion batch (multiplicity)")
@click.option("--cache", default=lambda: os.environ.get("BOLTZ_CACHE", str(Path("~/.boltz").expanduser())))
def warmup(max_seq, max_msa, n_samples, cache):
    """Pre-compile all ttnn kernels for Boltz-2 inference."""
    import gc

    from tt_boltz.tenstorrent import (
        WeightScope, PairformerModule, MSAModule, DiffusionModule,
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
    pf.load_state_dict(WeightScope.wrap(state).child("pairformer_module").as_dict(), strict=False)
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
    pf_tpl.load_state_dict(WeightScope.wrap(state).child("template_module.pairformer").as_dict(), strict=False)
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
    msa_mod.load_state_dict(WeightScope.wrap(state).child("msa_module").as_dict(), strict=False)
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
    diff_sd = WeightScope.wrap(state).child("structure_module.score_model").as_dict()
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
