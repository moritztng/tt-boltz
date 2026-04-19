#!/usr/bin/env python3
"""Benchmark tt-boltz against GPU baselines (Boltz-1, AlphaFold3, Chai-1).

Usage:
    python benchmark.py setup                                   # download dataset
    tt-boltz predict eval_data/queries/ --out_dir eval_data/     # predict
    python benchmark.py compare                                 # evaluate & compare

Dataset: 542 PDB targets from the Boltz-1 paper (Wohlwend et al. 2024).
Evaluation: OpenStructure (Docker), matching the paper's methodology.
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "eval_data"
QUERIES = DATA / "queries"
MSA = DATA / "msa"
TARGETS = DATA / "targets"
BASELINES = DATA / "gpu_results_test.csv"
PRED_DIR = DATA / "boltz_results_queries" / "structures"
EVAL_DIR = DATA / "evals"

GDRIVE_ID = "1JvHlYUMINOaqPTunI9wBYrfYniKgVmxf"
OST_IMAGE = "registry.scicore.unibas.ch/schwede/openstructure:2.11.1"

# Flags match the Boltz-1 paper (OpenStructure 2.8.0). The
# --lddt-no-stereochecks flag is required for OST 2.11.1 to
# reproduce the 2.8.0 behavior (2.8.0 had no stereo penalty).
OST_POLYMER = (
    "compare-structures "
    "-m {model} -r {ref} --fault-tolerant "
    "--min-pep-length 4 --min-nuc-length 4 "
    "--lddt --lddt-no-stereochecks --bb-lddt --qs-score --dockq "
    "--ics --ips --rigid-scores --patch-scores --tm-score "
    "-o {out}"
)
OST_LIGAND = (
    "compare-ligand-structures "
    "-m {model} -r {ref} --fault-tolerant "
    "--lddt-pli --rmsd --substructure-match -o {out}"
)

# Paper's Figure 1 metrics.
KEY_METRICS = ["lddt", "dockq_>0.23", "lddt_pli", "rmsd<2"]
ALL_METRICS = [
    "lddt", "bb_lddt", "tm_score",
    "dockq_>0.23", "dockq_>0.49",
    "lddt_pli", "rmsd<2", "rmsd<5",
]
METRIC_LABELS = {
    "lddt": "lDDT", "bb_lddt": "bb-lDDT", "tm_score": "TM-score",
    "dockq_>0.23": "DockQ >0.23", "dockq_>0.49": "DockQ >0.49",
    "lddt_pli": "lDDT-PLI", "rmsd<2": "RMSD <2A", "rmsd<5": "RMSD <5A",
}


# ── setup ──────────────────────────────────────────────────────────────────

def cmd_setup(args):
    """Download the Boltz-1 evaluation dataset and pull Docker image."""
    if QUERIES.exists() and TARGETS.exists() and BASELINES.exists():
        print(f"Dataset ready: {sum(1 for _ in QUERIES.glob('*.yaml'))} queries, "
              f"{sum(1 for _ in TARGETS.iterdir())} targets")
    else:
        _download_dataset(args)

    _fix_msa_paths()

    _check_docker(pull=True)

    print(f"\nSetup complete. Next:")
    print(f"  tt-boltz predict {QUERIES}/ --out_dir {DATA}/")
    print(f"  python benchmark.py compare")


def _download_dataset(args):
    DATA.mkdir(parents=True, exist_ok=True)
    zip_path = DATA / "boltz_results_final.zip"

    if not zip_path.exists():
        print("Downloading Boltz evaluation dataset (~10 GB)...")
        try:
            subprocess.run(["gdown", GDRIVE_ID, "-O", str(zip_path)], check=True)
        except FileNotFoundError:
            sys.exit(
                "gdown not found. Install: pip install gdown\n"
                f"Or download manually from:\n"
                f"  https://drive.google.com/file/d/{GDRIVE_ID}/view\n"
                f"Save to: {zip_path}\nThen re-run: python benchmark.py setup"
            )

    print("Extracting...")
    import zipfile
    extract = {
        "boltz_results_final/inputs/test/boltz/msa/": MSA,
        "boltz_results_final/inputs/test/boltz/": QUERIES,
        "boltz_results_final/targets/test/": TARGETS,
    }
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member == "boltz_results_final/results_test.csv":
                with zf.open(member) as src:
                    BASELINES.write_bytes(src.read())
                continue
            for prefix, dst in extract.items():
                if member.startswith(prefix) and not member.endswith("/"):
                    dst.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src:
                        (dst / Path(member).name).write_bytes(src.read())
                    break

    if not args.keep_zip:
        zip_path.unlink()
        print("Deleted zip.")


def _fix_msa_paths():
    """Rewrite MSA paths in YAMLs to point to eval_data/msa/."""
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML required: pip install pyyaml")

    print("Fixing MSA paths...", end=" ", flush=True)
    msa_str = str(MSA.resolve())
    fixed = 0
    for yf in sorted(QUERIES.glob("*.yaml")):
        with open(yf) as f:
            data = yaml.safe_load(f)
        changed = False
        for seq in data.get("sequences", []):
            # msa is inside protein/dna/rna sub-dict, not at seq level
            for chain_type in ("protein", "dna", "rna"):
                chain = seq.get(chain_type)
                if not chain:
                    continue
                m = chain.get("msa")
                if isinstance(m, str) and not m.startswith(msa_str):
                    chain["msa"] = str(MSA / Path(m).name)
                    changed = True
                elif isinstance(m, dict):
                    for k in m:
                        if isinstance(m[k], str) and not m[k].startswith(msa_str):
                            m[k] = str(MSA / Path(m[k]).name)
                            changed = True
        if changed:
            fixed += 1
            with open(yf, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"{fixed} files updated.")


def _check_docker(pull=False):
    r = subprocess.run(["docker", "info"], capture_output=True)
    if r.returncode != 0:
        msg = "Docker not available. Install and add yourself to the docker group:\n"
        msg += "  sudo usermod -aG docker $USER && newgrp docker"
        if pull:
            print(f"Warning: {msg}")
        else:
            sys.exit(msg)
        return
    if pull:
        print(f"Pulling {OST_IMAGE}...", end=" ", flush=True)
        r = subprocess.run(["docker", "pull", OST_IMAGE], capture_output=True)
        print("ok." if r.returncode == 0 else "failed (non-fatal).")


# ── compare ────────────────────────────────────────────────────────────────

def cmd_compare(args):
    """Evaluate predictions with OpenStructure and compare to GPU baselines."""
    if not PRED_DIR.exists() or not list(PRED_DIR.glob("*.cif")):
        sys.exit(
            f"No predictions found at {PRED_DIR}\n"
            f"Run: tt-boltz predict {QUERIES}/ --out_dir {DATA}/"
        )
    if not TARGETS.exists():
        sys.exit(f"No targets at {TARGETS}. Run: python benchmark.py setup")

    preds = {p.stem: p for p in PRED_DIR.glob("*.cif")}
    targets = {
        t.name.replace(".cif.gz", "").replace(".cif", ""): t
        for t in TARGETS.iterdir() if t.name.endswith(".cif") or t.name.endswith(".cif.gz")
    }
    common = sorted(set(preds) & set(targets))
    if not common:
        sys.exit("No overlap between predictions and targets.")

    _check_docker()

    # Run OpenStructure (idempotent — skips existing JSONs)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    to_run = [n for n in common if not (EVAL_DIR / f"{n}.json").exists()]
    if to_run:
        print(f"Evaluating {len(to_run)} targets ({len(common) - len(to_run)} cached)...")
        mount = str(ROOT.resolve())
        _eval_one(to_run[0], preds[to_run[0]], targets[to_run[0]], mount)
        if len(to_run) > 1:
            with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
                futs = [pool.submit(_eval_one, n, preds[n], targets[n], mount)
                        for n in to_run[1:]]
                for i, _ in enumerate(concurrent.futures.as_completed(futs), 2):
                    if i % 20 == 0 or i == len(to_run):
                        print(f"  {i}/{len(to_run)}")

    # Parse and display
    tt = _aggregate(common)
    if tt.empty:
        sys.exit("No evaluation results. Check Docker permissions.")

    _print_results(tt, common)


def _eval_one(name, pred, ref, mount):
    for template, suffix in [(OST_POLYMER, ""), (OST_LIGAND, "_ligand")]:
        out = EVAL_DIR / f"{name}{suffix}.json"
        if out.exists():
            continue
        cmd = template.format(model=pred, ref=ref, out=out)
        full = f"docker run --rm -u $(id -u):$(id -g) -v {mount}:{mount} {OST_IMAGE} {cmd}"
        subprocess.run(full, shell=True, check=False, capture_output=True)


def _aggregate(names):
    """Parse OpenStructure JSONs into a tidy DataFrame."""
    rows = []
    for name in names:
        m = _parse_one(name)
        if m:
            for metric, val in m.items():
                rows.append({"target": name, "metric": metric, "value": val})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _parse_one(name):
    """Parse polymer + ligand JSONs for one target. Returns metrics dict."""
    metrics = {}

    poly = EVAL_DIR / f"{name}.json"
    if poly.exists():
        d = json.load(open(poly))
        for k in ["lddt", "bb_lddt", "tm_score", "rmsd"]:
            if k in d:
                metrics[k] = d[k]
        if d.get("dockq"):
            valid = [v for v in d["dockq"] if v is not None]
            if valid:
                metrics["dockq_>0.23"] = np.mean([float(v > 0.23) for v in valid])
                metrics["dockq_>0.49"] = np.mean([float(v > 0.49) for v in valid])

    lig = EVAL_DIR / f"{name}_ligand.json"
    if lig.exists():
        d = json.load(open(lig))
        if "lddt_pli" in d:
            s = [x["score"] for x in d["lddt_pli"].get("assigned_scores", [])]
            s.extend([0] * len(d["lddt_pli"].get("model_ligand_unassigned_reason", {})))
            if s:
                metrics["lddt_pli"] = np.mean(s)
        if "rmsd" in d:
            s = [x["score"] for x in d["rmsd"].get("assigned_scores", [])]
            s.extend([100] * len(d["rmsd"].get("model_ligand_unassigned_reason", {})))
            if s:
                metrics["rmsd<2"] = np.mean([x < 2.0 for x in s])
                metrics["rmsd<5"] = np.mean([x < 5.0 for x in s])

    return metrics or None


def _print_results(tt, targets):
    target_set = set(targets)
    gpu = pd.read_csv(BASELINES) if BASELINES.exists() else None
    tools = [("boltz", "Boltz-1"), ("af3", "AF3"), ("chai", "Chai-1")]

    w = 14  # column width
    print(f"\n{'=' * 74}")
    print(f"  Benchmark: Boltz PDB Test Set ({len(target_set)} / 542 targets)")
    print(f"  Evaluation: OpenStructure 2.11.1 (--lddt-no-stereochecks)")
    print(f"{'=' * 74}")

    hdr = f"  {'':16} {'TT-Boltz-2':>{w}}"
    if gpu is not None:
        for _, label in tools:
            hdr += f" {label:>{w}}"
    hdr += f"  {'n':>4}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for metric in ALL_METRICS:
        tt_m = tt[(tt["metric"] == metric) & (tt["target"].isin(target_set))]
        if tt_m.empty:
            continue
        label = METRIC_LABELS.get(metric, metric)
        star = "*" if metric in KEY_METRICS else " "
        line = f" {star}{label:>15} {tt_m['value'].mean():>{w}.4f}"

        if gpu is not None:
            for tool_id, _ in tools:
                g = gpu[(gpu["tool"] == tool_id) & (gpu["metric"] == metric)
                        & (gpu["target"].isin(target_set))]
                if not g.empty:
                    line += f" {g['top1'].mean():>{w}.4f}"
                else:
                    line += f" {'--':>{w}}"
        line += f"  {len(tt_m):>4}"
        print(line)

    print()
    print("  * = key metrics from the Boltz-1 paper (Figure 1)")
    print(f"  TT-Boltz-2: 1 sample | GPU baselines: top-1 of 5 samples")

    out = EVAL_DIR / "tt_boltz_results.csv"
    tt.to_csv(out, index=False)
    print(f"\n  Per-target results saved to: {out}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("setup", help="Download dataset and pull Docker image")
    s.add_argument("--keep-zip", action="store_true")

    c = sub.add_parser("compare", help="Evaluate predictions and compare to baselines")
    c.add_argument("--workers", type=int, default=16, help="Parallel Docker workers")

    args = p.parse_args()
    if args.cmd == "setup":
        cmd_setup(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

