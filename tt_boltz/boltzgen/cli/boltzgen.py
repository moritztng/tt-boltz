#!/usr/bin/env python3
"""BoltzGen pipeline CLI (Tenstorrent-only build).

Two halves:

1. ``configure`` resolves the per-step YAML configs and writes an
   ``OUTPUT/config/<step>.yaml`` for each enabled pipeline step, plus a
   manifest ``OUTPUT/steps.yaml`` listing them in execution order.

2. ``execute`` reads the manifest and runs each step in-process by
   instantiating the YAML config and calling ``Task.run(...)``. Steps share
   the Tenstorrent device handle and program cache across the whole
   pipeline — spawning subprocesses would discard both and re-pay model
   load per step.

``run`` configures and executes in one shot.

Pipeline tasks live alongside this file:
    * Predict — tt_boltz/boltzgen/task/predict/predict.py
    * Analyze — tt_boltz/boltzgen/task/analyze/analyze.py
    * Filter  — tt_boltz/boltzgen/task/filter/filter.py
"""

# Select the Tenstorrent device(s) BEFORE importing anything under
# ``tt_boltz.boltzgen`` — its package __init__ pulls in the ttnn adapters,
# and ttnn reads TT_VISIBLE_DEVICES at import time. Mirrors how tt-boltz's
# worker processes set TT_VISIBLE_DEVICES before importing ttnn so the
# chosen physical chip becomes logical device 0 (the one get_device() opens).
import os as _os
import sys as _sys


def _apply_device_selection_from_argv(argv: list[str]) -> None:
    """Honour ``--device_ids`` (e.g. ``--device_ids 2``) before ttnn loads.

    boltzgen runs a single logical TT device per process, so the value is
    used as TT_VISIBLE_DEVICES: the selected chip(s) become the only ones
    ttnn enumerates, and get_device() opens logical device 0 among them.
    An explicit TT_VISIBLE_DEVICES already in the environment wins.
    """
    value = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--device_ids":
            value = argv[i + 1] if i + 1 < len(argv) else None
            i += 2
            continue
        if arg.startswith("--device_ids="):
            value = arg.split("=", 1)[1]
        i += 1
    if value is None or value == "":
        return
    if _os.environ.get("TT_VISIBLE_DEVICES"):
        return
    _os.environ["TT_VISIBLE_DEVICES"] = value


_apply_device_selection_from_argv(_sys.argv[1:])

from tt_boltz.boltzgen.utils.quiet import quiet_startup

quiet_startup()

import collections
import argparse
import copy
from dataclasses import dataclass
import os
import subprocess
import time
import math
import re
import shutil
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml
import torch

from tt_boltz.boltzgen._config import (
    deep_merge as _deep_merge,
    dotlist_to_dict as _dotlist_to_dict,
    instantiate as _instantiate,
    load_yaml as _load_yaml,
    resolve_interpolations as _resolve_interpolations,
    save_yaml as _save_yaml,
)

from tt_boltz.data import const
from tt_boltz.data.mol import load_canonicals
from tt_boltz.boltzgen.data.parse.schema import YamlDesignParser
from tt_boltz.boltzgen.data.write_mmcif import to_mmcif
from tt_boltz.boltzgen.task.task import Task
from importlib.metadata import PackageNotFoundError, version as pkg_version

### Paths and constants ####
# Get the path to the project root (where main.py and configs/ are located)
# Since we're now in src/boltzgen/, we need to go up 3 levels
path_to_script = Path(__file__)
project_root = (
    path_to_script.parent.parent
)  # Go up from src/boltzgen/cli to project root
config_dir = project_root / "resources/config"

step_names = [
    "design",
    "inverse_folding",
    "design_folding",
    "folding",
    "affinity",
    "analysis",
    "filtering",
]

### Protocol-specific configuration overrides (which can be overridden by user) ####
protocol_configs = {
    "protein-anything": {},  # base config corresponds to protein-anything
    "peptide-anything": {
        # Note that in inverse folding step we also avoid cysteines by default; this is implemented elsewhere.
        "analysis": ["largest_hydrophobic=false", "largest_hydrophobic_refolded=false"],
        "filtering": [
            "filter_cysteine=true",
            "alpha=0.01",
            "refolding_rmsd_threshold=2",
        ],
    },
    "protein-small_molecule": {
        "analysis": ["affinity_metrics=true"],
        "filtering": ["use_affinity=true"],
    },
    "nanobody-anything": {
        "analysis": ["largest_hydrophobic=false", "largest_hydrophobic_refolded=false"],
        "filtering": ["filter_cysteine=true"],
    },
    "antibody-anything": {
        "analysis": ["largest_hydrophobic=false", "largest_hydrophobic_refolded=false"],
        "filtering": ["filter_cysteine=true"],
    },
    "protein-redesign": {
        # For redesigning/optimizing existing proteins (e.g., symmetric dimers)
        # where all chains may have designed residues. Skips design_folding and
        # uses design_mask (not chain_design_mask) for target/template definition.
        "folding": ["data.design_mask_templates=true"],
        "analysis": ["use_design_mask_for_target=true"],
        "filtering": [
            "metrics_override={design_to_target_iptm: null, neg_min_design_to_target_pae: null, design_ptm: null, plip_hbonds_refolded: null, plip_saltbridge_refolded: null, delta_sasa_refolded: null, plip_hbonds: null, plip_saltbridge: null, delta_sasa_original: null, design_residue_iptm: 1, iptm: 2, ptm: 3, neg_filter_rmsd_design: 4}",
        ],
    },
}
assert all(
    step_name in step_names for cfg in protocol_configs.values() for step_name in cfg
)


### Model checkpoints and other artifacts ####
# All BoltzGen weights + mols are mirrored on the same GCP bucket as the
# tt-boltz core artifacts and fetched directly over HTTPS.
BOLTZGEN_ARTIFACTS_URL = "https://storage.googleapis.com/tt-boltz-artifacts/boltzgen"
ARTIFACTS: dict[str, tuple[str, str]] = {
    "design-diverse":   (f"{BOLTZGEN_ARTIFACTS_URL}/boltzgen1_diverse.ckpt",   "model"),
    "design-adherence": (f"{BOLTZGEN_ARTIFACTS_URL}/boltzgen1_adherence.ckpt", "model"),
    "inverse-fold":     (f"{BOLTZGEN_ARTIFACTS_URL}/boltzgen1_ifold.ckpt",     "model"),
    "folding":          (f"{BOLTZGEN_ARTIFACTS_URL}/boltz2_conf_final.ckpt",   "model"),
    "affinity":         (f"{BOLTZGEN_ARTIFACTS_URL}/boltz2_aff.ckpt",          "model"),
    "moldir":           (f"{BOLTZGEN_ARTIFACTS_URL}/mols.zip",                 "dataset"),
}


### CLI arguments ###
def add_configure_arguments(
    parser: argparse.ArgumentParser, *, output_required: bool = False
) -> None:
    # General configuration options
    p = parser.add_argument_group("general configuration")
    p.add_argument(
        "--protocol",
        type=str,
        choices=list(protocol_configs.keys()),
        default="protein-anything",
        help="Protocol to use for the design. This determines default settings and in some cases what steps "
        "are run. Default: %(default)s",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=output_required,
        help="Output directory for pipeline results",
    )
    p.add_argument(
        "--config",
        nargs="+",
        action="append",
        help="Override pipeline step configuration, in format <step_name> <arg1>=<value1> <arg2>=<value2> ..."
        "(example: '--config folding num_workers=4 trainer.devices=4'). Can be used multiple times.",
    )
    p.add_argument(
        "--devices",
        type=int,
        help="Number of devices to use. Default is all devices available.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        help="Number of DataLoader worker processes.",
        default=1,
    )
    p.add_argument(
        "--config_dir",
        type=Path,
        help=f"Path to the directory of default config files. Default: %(default)s",
        default=config_dir,
    )
    p.add_argument(
        "--use_kernels",
        help="Whether to use kernels. One of 'auto', 'true', or 'false'. Default: %(default)s. "
        "If 'auto', will use kernels if the device capability is >= 8.",
        choices=["auto", "true", "false"],
        default="auto",
    )
    p.add_argument(
        "--moldir",
        type=str,
        help="Path to the moldir. Default: %(default)s",
        default=ARTIFACTS["moldir"][0],
    )
    p.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing results across all steps. Generate only as many new designs are "
        "needed to achieve the specified total number of designs.",
    )

    # Design configuration options
    p = parser.add_argument_group("design")
    p.add_argument(
        "--num_designs",
        type=int,
        help="Number of total designs to generate. This commonly would be something like 10,000"
        "After generating 10,000 designs we then filter down to --budget many designs in the filter step",
        default=10000,
    )
    p.add_argument(
        "--diffusion_batch_size",
        type=int,
        default=None,
        help="Number of diffusion samples to generate per trunk run. If not specified, "
        "this defaults to 1 if --num-designs is less than 100, and 10 otherwise. Note that "
        "for design tasks that randomly sample the binder length (or use randomness in other "
        "ways), all designs generated in the same batch will share the same length. "
        "Having a large diffusion batch size compared to the total number of designs to "
        "generate will therefore not evenly sample the possible lengths.",
    )
    p.add_argument(
        "--design_checkpoints",
        type=str,
        nargs="+",
        help="Path to the boltzgen checkpoint(s). One or more checkpoints are supported. Just specifying an individual path here will work."
        "Each will be used for an equal fraction of the designs. By default, two checkpoints are used. "
        "Default: %(default)s",
        default=[
            ARTIFACTS["design-diverse"][0],
            ARTIFACTS["design-adherence"][0],
        ],
    )
    p.add_argument(
        "--step_scale",
        type=str,
        help="Fixed step scale to use (e.g. 1.8). Default is to use a schedule",
        default=None,
    )
    p.add_argument(
        "--noise_scale",
        type=str,
        help="Fixed noise scale to use (e.g. 0.98). Default is to use a schedule",
        default=None,
    )

    # Inverse folding configuration options
    p = parser.add_argument_group("inverse folding")
    p.add_argument(
        "--skip_inverse_folding",
        action="store_true",
        help="Skip inverse folding step",
    )
    p.add_argument(
        "--inverse_fold_num_sequences",
        type=int,
        help="Number of sequences per backbone to generate in the inverse fold step. Default: %(default)s",
        default=1,
    )
    p.add_argument(
        "--inverse_fold_checkpoint",
        type=str,
        help="Path to the inverse fold checkpoint. Default: %(default)s",
        default=ARTIFACTS["inverse-fold"][0],
    )
    p.add_argument(
        "--inverse_fold_avoid",
        type=str,
        default=None,
        help="Disallowed residues as a string of one letter amino acid codes, e.g. 'KEC'. "
        "This is implemented at the inverse fold step, so it only affects results if inverse folding is "
        "enabled. Default: none for protein design, 'C' for peptide and antibody/nanobody design. Pass an empty list if you want Cysteins to be generated if you are using antibody/nanobody/peptide protocol",
    )
    p.add_argument(
        "--only_inverse_fold",
        action="store_true",
        help="Skip design step and only run inverse folding. Requires a fully specified structure.",
    )

    # Folding and affinity prediction configuration options
    p = parser.add_argument_group("folding and affinity prediction")
    p.add_argument(
        "--folding_checkpoint",
        type=str,
        help="Path to the folding checkpoint. Default: %(default)s",
        default=ARTIFACTS["folding"][0],
    )
    p.add_argument(
        "--affinity_checkpoint",
        type=str,
        help="Path to the affinity predictor checkpoint. Default: %(default)s",
        default=ARTIFACTS["affinity"][0],
    )

    # Filtering configuration options
    p = parser.add_argument_group("filtering")
    p.add_argument(
        "--budget",
        type=int,
        help="How many designs should be in the final diversity optimized set. This is used in the filtering step.",
        default=30,
    )
    p.add_argument(
        "--alpha",
        type=float,
        help="Trade-off for sequence diversity selection: 0.0=quality-only, 1.0=diversity-only. Default is "
        "0.01 (peptide-anything protocol) or 0.001 (other protocols).",
        default=None,
    )
    p.add_argument(
        "--filter_biased",
        choices=["true", "false"],
        help="Remove amino-acid composition outliers (default caps on ALA/GLY/GLU/LEU/VAL). Default: %(default)s.",
        default="true",
    )
    p.add_argument(
        "--metrics_override",
        nargs="+",
        help="Per-metric inverse-importance weights for ranking. "
        "Format: metric_name=weight (e.g., plip_hbonds_refolded=4 delta_sasa_refolded=2). "
        "A larger value down-weights that metric's rank. Use 'metric_name=none' to remove a metric.",
        default=None,
    )
    p.add_argument(
        "--additional_filters",
        nargs="+",
        help="Extra hard filters. Format: feature>threshold or feature<threshold "
        "(e.g., 'design_ALA>0.3' 'design_GLY<0.2'). Use '>' if higher is better, '<' if lower is better. "
        "Make sure to single-quote the strings so your shell doesn't get confused by < and > characters.",
        default=None,
    )
    p.add_argument(
        "--size_buckets",
        nargs="+",
        help="Optional constraint for maximum number of designs in size ranges. "
        "Format: min-max:count (e.g., 10-20:5 20-30:10 30-40:5).",
        default=None,
    )
    p.add_argument(
        "--refolding_rmsd_threshold",
        type=float,
        help="Threshold used for RMSD-based filters (lower is better).",
        default=None,
    )


def add_models_download_options(p: argparse.ArgumentParser) -> None:
    p = p.add_argument_group("model and data download options")
    p.add_argument(
        "--force_download",
        help="Force a (re)-download of models and data.",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--cache",
        type=Path,
        help="Directory where downloaded models will be stored. Default: ~/.cache",
        default=None,
    )


def add_device_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--device_ids",
        type=str,
        default=None,
        metavar="IDS",
        help="Tenstorrent device ID(s) to run on. A single ID (e.g. '2') runs on "
        "that chip (made the only one ttnn sees, via TT_VISIBLE_DEVICES). A "
        "comma-separated list (e.g. '0,1,2,3') splits the designs evenly across "
        "those cards and merges the results — the output is identical in layout "
        "to a single-device run. Default: device 0 / TT_VISIBLE_DEVICES if set.",
    )


def add_execute_core_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--steps",
        nargs="+",
        choices=step_names,
        help="Run only the specified pipeline steps (default: run all steps)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: no Rich display, no output suppression",
    )
    p.add_argument(
        "--log",
        action="store_true",
        help="With --debug: print per-stage progress lines",
    )


def build_run_parser(subparsers) -> argparse.ArgumentParser:
    run_parser = subparsers.add_parser(
        "run",
        description="Boltzgen binder design pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Run the binder design pipeline",
        epilog=__doc__,
    )
    group = run_parser.add_argument_group("design specification")
    group.add_argument(
        "design_spec",
        nargs="+",
        type=Path,
        help="Path(s) to design specification YAML file(s), or a directory containing prepared configs",
    )
    add_configure_arguments(run_parser, output_required=False)
    add_execute_core_arguments(run_parser)
    add_device_arguments(run_parser)
    add_models_download_options(run_parser)
    return run_parser


def build_execute_parser(subparsers) -> argparse.ArgumentParser:
    execute_parser = subparsers.add_parser(
        "execute",
        description="Execute a pre-configured pipeline from a directory of config files",
        help="Run pipeline from pre-generated configuration files",
    )
    execute_parser.add_argument(
        "output",
        type=Path,
        help="Directory containing pre-configured pipeline files (generated by 'configure' command)",
    )
    add_execute_core_arguments(execute_parser)
    add_device_arguments(execute_parser)
    return execute_parser


def build_configure_parser(subparsers) -> argparse.ArgumentParser:
    configure_parser = subparsers.add_parser(
        "configure",
        description="Generate resolved pipeline configuration files without executing steps",
        help="Create configuration files for later execution",
    )
    group = configure_parser.add_argument_group("design specification")
    group.add_argument(
        "design_spec",
        nargs="+",
        type=Path,
        help="Path(s) to design specification YAML file(s)",
    )

    group = configure_parser.add_argument_group("steps to configure")
    group.add_argument(
        "--steps",
        nargs="+",
        choices=step_names,
        help="Configure only the specified pipeline steps (default: all steps)",
    )

    add_configure_arguments(configure_parser, output_required=True)
    add_device_arguments(configure_parser)
    add_models_download_options(configure_parser)

    return configure_parser


def build_download_parser(subparsers) -> argparse.ArgumentParser:
    download_parser = subparsers.add_parser(
        "download",
        help="Download boltzgen model weights and supporting assets",
    )
    group = download_parser.add_argument_group(
        "artifacts to download (positional argument)"
    )
    group.add_argument(
        "artifacts",
        nargs="+",
        default=[],
        choices=sorted(ARTIFACTS.keys()) + ["all"],
        help="Subset of artifacts to download, or 'all' to download all artifacts.",
    )
    add_models_download_options(download_parser)
    return download_parser


def build_check_parser(subparsers) -> argparse.ArgumentParser:
    check_parser = subparsers.add_parser(
        "check",
        description="Check design specification files for validity and optionally output mmCIF",
        help="Validate design specification files",
    )
    check_parser.add_argument(
        "design_spec",
        nargs="+",
        type=Path,
        help="Path(s) to design specification YAML file(s)",
    )
    check_parser.add_argument(
        "--output",
        type=Path,
        help="Output directory to write mmCIF files (optional)",
    )
    check_parser.add_argument(
        "--moldir",
        type=str,
        help="Path to the moldir. Default: %(default)s",
        default=ARTIFACTS["moldir"][0],
    )

    add_models_download_options(check_parser)
    return check_parser


def build_merge_parser(subparsers) -> argparse.ArgumentParser:
    merge_parser = subparsers.add_parser(
        "merge",
        description="Merge multiple BoltzGen output directories so filtering can be rerun on the combined set.",
        help="Combine finished pipeline outputs into a single directory",
    )
    merge_parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="Paths to completed BoltzGen output directories (results of 'run' or 'execute')",
    )
    merge_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination directory for the merged outputs",
    )
    merge_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignored: kept temporarily for backwards compatibility. In all cases, the destination data is overwritten.",
    )
    return merge_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boltzgen",
        description="Boltzgen command line interface",
    )
    # Support: boltzgen -v / --version
    def get_package_version() -> str:
        try:
            return pkg_version("boltzgen")
        except PackageNotFoundError:
            return "unknown"
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"boltzgen {get_package_version()}",
        help="Print version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_run_parser(subparsers)
    build_configure_parser(subparsers)
    build_execute_parser(subparsers)
    build_download_parser(subparsers)
    build_check_parser(subparsers)
    build_merge_parser(subparsers)
    return parser


#### Commands ####
def _device_id_list(device_ids: str | None) -> list[int]:
    """Parse ``--device_ids`` (e.g. '0,2,3') into a list of TT device IDs."""
    if not device_ids:
        return []
    return [int(d) for d in (p.strip() for p in device_ids.split(",")) if d]


def _rewrite_run_argv(argv: list[str], strip: set[str], additions: list[str]) -> list[str]:
    """Drop ``strip`` options (and their values) from a ``run`` argv, then append
    ``additions``. Positionals (design specs) come before options, so skipping
    value tokens up to the next ``-`` handles both single- and multi-value opts."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.split("=", 1)[0] in strip:
            i += 1
            if "=" not in tok:
                while i < len(argv) and not argv[i].startswith("-"):
                    i += 1
            continue
        out.append(tok)
        i += 1
    return out + additions


def run_command(args: argparse.Namespace) -> None:
    """
    Run the **complete binder design pipeline** end-to-end by running:
        1. `configure_command(args)` – generates the per-step YAML configurations.
        2. `execute_command(args)` – launches each pipeline step based on those YAMLs.

    Typical CLI usage:
        $ boltzgen run path/to/design.yaml --output out_dir --protocol protein-anything

    With several TT devices (``--device_ids 0,1,2,3``) the designs are split
    evenly across the cards: one single-device run per card on its own shard,
    then the shards are merged and filtered once so the final output directory
    is identical in layout to a single-device run.
    """
    # Validate required arguments for running from design specs
    if not args.output:
        print("No output directory specified. Exiting.")
        return

    devices = _device_id_list(args.device_ids)
    if len(devices) > 1:
        _run_multi_device(args, devices)
        return

    print("\n=== Configuring pipeline ===")
    configure_command(args)

    print("\n=== Executing pipeline ===")
    execute_command(args)


def _run_multi_device(args: argparse.Namespace, devices: list[int]) -> None:
    """Fan the design set out across ``devices``, then merge + filter once."""
    n = len(devices)
    total = args.num_designs
    # Split designs as evenly as possible; the first few cards take the remainder.
    counts = [total // n + (1 if i < total % n else 0) for i in range(n)]

    output = args.output
    shard_root = output / "shards"
    base_argv = sys.argv[1:]  # the user's "run <spec> ..." argv

    print(f"\n=== Distributing {total} designs across {n} devices {devices} ===")
    for dev, count in zip(devices, counts):
        print(f"  device {dev}: {count} designs -> {shard_root / f'device_{dev}'}")

    procs = []
    for dev, count in zip(devices, counts):
        shard_dir = shard_root / f"device_{dev}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        argv = _rewrite_run_argv(
            base_argv,
            strip={"--output", "--num_designs", "--device_ids"},
            additions=["--output", str(shard_dir),
                       "--num_designs", str(count),
                       "--device_ids", str(dev)],
        )
        # Pin the worker to its physical chip up front (TT_VISIBLE_DEVICES is read
        # at ttnn import); don't let the parent's multi-id value leak in.
        env = {**os.environ, "TT_VISIBLE_DEVICES": str(dev)}
        log_file = open(shard_dir / "run.log", "w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "tt_boltz.boltzgen.cli.boltzgen", *argv],
            stdout=log_file, stderr=subprocess.STDOUT, env=env,
        )
        procs.append({"dev": dev, "dir": shard_dir, "proc": proc, "log": log_file})

    _wait_for_shards(procs)

    failed = [p["dev"] for p in procs if p["proc"].returncode != 0]
    if failed:
        raise RuntimeError(
            f"Design shard(s) on device(s) {failed} failed; see "
            f"{shard_root}/device_<id>/run.log. Re-run with --reuse to resume."
        )

    # Global reduce: combine all shards, then filter once over the union so the
    # diversity-optimized final set is selected across every design.
    print(f"\n=== Merging {n} shards into {output} ===")
    merge_command(argparse.Namespace(
        sources=[p["dir"] for p in procs], output=output, overwrite=False))

    if args.steps and "filtering" not in args.steps:
        print("Skipping global filtering (not requested in --steps).")
        return
    print("\n=== Filtering merged designs ===")
    filter_args = copy.copy(args)
    filter_args.steps = ["filtering"]
    filter_args.reuse = True
    configure_command(filter_args)
    execute_command(filter_args)


def _wait_for_shards(procs: list[dict]) -> None:
    """Block until all shard processes exit, printing periodic progress."""
    def _count(d: Path, pattern: str) -> int:
        sub = d / "intermediate_designs"
        return len(list(sub.glob(pattern))) if sub.exists() else 0

    while any(p["proc"].poll() is None for p in procs):
        time.sleep(30)
        status = "  ".join(
            f"dev{p['dev']}:{'done' if p['proc'].poll() == 0 else ('FAILED' if p['proc'].poll() else 'run')}"
            f"({_count(p['dir'], 'input_*.cif')} designs)"
            for p in procs
        )
        print(f"[shards] {status}", flush=True)
    for p in procs:
        p["log"].close()


def download_command(args: argparse.Namespace) -> list[Path]:
    """
    Download **BoltzGen model checkpoints and the molecules directory** from the GCP bucket.

    Parameters
    ----------
    args.artifacts : list[str]
        List of artifact keys to download, or `["all"]` to fetch all available assets.
    args.cache : Path
        Cache directory for storing downloaded artifacts.

    Returns
    -------
    list[Path]
        Local file paths of successfully downloaded artifacts.

    Usually this is executed by `boltzgen run ...` but it can be used like:
        $ boltzgen download all
        $ boltzgen download design-diverse inverse-fold
    """
    selections = sorted(set(args.artifacts))
    if "all" in selections:
        selections = list(ARTIFACTS.keys())

    download_paths: list[Path] = []
    for name in selections:
        artifact, repo_type = ARTIFACTS[name]
        resolved_path = get_artifact_path(args, artifact, repo_type=repo_type)
        print(f"Downloading {name} to {resolved_path}")
        download_paths.append(resolved_path)

    return download_paths


def configure_command(args: argparse.Namespace) -> None:
    """
    Generate **resolved per-step YAML configuration files** for the binder-design pipeline.

    This command constructs a `BinderDesignPipeline` from user and protocol parameters,
    validates design specifications, and writes out all configuration files required for running the pipeline. It does NOT run the pipeline.

    Outputs
    -------
    * `<output_dir>/config/<step>.yaml` — fully resolved config for each pipeline step.
    * `<output_dir>/steps.yaml` — manifest of steps and config file paths.

    This stage does **not execute** any model code; it only prepares the YAMLs used by
    `execute_command(...)`.

    Usually this is executed by `boltzgen run ...` but it can be used like:
        $ boltzgen configure path/to/design.yaml --output out_dir --protocol peptide-anything
    """
    moldir = get_artifact_path(args, args.moldir, repo_type="dataset")
    mols = load_canonicals(moldir=moldir)

    # Setup output directory
    output_dir = args.output
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {output_dir}")
    else:
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True)

    check_design_specs(args, moldir, mols)

    pipeline = BinderDesignPipeline(args, moldir)
    pipeline.pretty_print()

    # Check that tasks can be instantiated
    for step in pipeline.steps:
        step.check()

    # Make the config subdir in output
    config_dir = output_dir / "config"
    if config_dir.exists():
        # Rename it to be previous-config-XXX
        counter = 1
        while config_dir.with_name(f"previous-config-{counter}").exists():
            counter += 1
        prev_config_dir = config_dir.with_name(f"previous-config-{counter}")
        print(f"Renaming existing config directory to {prev_config_dir}")
        config_dir.rename(prev_config_dir)
    config_dir.mkdir(parents=True)

    # Prepare step configurations and collect step info
    steps_info = []
    for step in pipeline.steps:
        config = step.get_config()
        config_filename = f"{step.name}.yaml"
        config_path = config_dir / config_filename
        _save_yaml(config, config_path)

        # Add step info for steps.yaml (use relative path from output directory)
        steps_info.append(
            {"name": step.name, "config_file": str(config_path.relative_to(output_dir))}
        )

    # Write steps.yaml file
    steps_yaml_path = output_dir / "steps.yaml"
    steps_data = {"steps": steps_info}

    with open(steps_yaml_path, "w") as f:
        yaml.dump(steps_data, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration complete. Configs written to {config_dir}")
    print(f"Steps manifest written to {steps_yaml_path}")


def check_command(args: argparse.Namespace) -> None:
    """
    Validate **design specification YAML files** and write cif file for visualization.

    This command parses input design specs using `YamlDesignParser`, verifies structure
    integrity, highlights unresolved residues, and  outputs colored mmCIF visualizations.

    Parameters
    ----------
    args.design_spec : list[Path]
        Input YAML file(s) describing the binder design.
    args.output : Path, optional
        Directory to write mmCIF outputs (optional).
    args.moldir : str
        Path or Hugging Face reference to the molecule dataset.

    Typical CLI usage:
        $ boltzgen check path/to/design.yaml --output checked/

    This function does **not** execute the pipeline — it only validates inputs.
    """
    moldir = get_artifact_path(args, args.moldir, repo_type="dataset")
    mols = load_canonicals(moldir=moldir)

    if args.output:
        output_dir = args.output
        if output_dir.exists():
            if not output_dir.is_dir():
                raise ValueError(
                    f"Output path exists and is not a directory: {output_dir}"
                )
        else:
            print(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True)

    check_design_specs(args, moldir, mols)


def execute_command(args: argparse.Namespace) -> None:
    """
    Execute a **pre-configured binder design pipeline** from a directory of YAML files.

    Reads the `steps.yaml` manifest written by `configure_command(...)` and runs
    each step sequentially in-process (sharing the Tenstorrent device handle and
    program cache across the whole pipeline).

    Options
    --------
    * `--steps` : Restrict to specific pipeline steps.
    * `--reuse` : Skip recomputation for existing results.

    Expected directory structure (produced by `configure_command(...)`):
        output_dir/
            ├── config/
            │     ├── design.yaml
            │     ├── folding.yaml
            │     └── ...
            └── steps.yaml

    Usually this is executed by `boltzgen run ...` but it can be used like:
        $ boltzgen execute --output out_dir
    """
    config_dir = args.output

    if not config_dir.exists() or not config_dir.is_dir():
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

    # Look for steps.yaml file
    steps_yaml_path = config_dir / "steps.yaml"
    if not steps_yaml_path.exists():
        raise FileNotFoundError(
            f"Steps manifest not found: {steps_yaml_path}. Run 'boltzgen configure' first."
        )

    # Load steps from steps.yaml
    with open(steps_yaml_path, "r") as f:
        steps_data = yaml.safe_load(f)

    if not isinstance(steps_data, dict) or "steps" not in steps_data:
        raise ValueError(f"Invalid steps.yaml format in {steps_yaml_path}")

    # Filter steps if specific steps are requested
    enabled_steps = set(args.steps) if args.steps else None
    resolved_steps: List[Tuple[str, Path]] = []

    for step_info in steps_data["steps"]:
        step_name = step_info["name"]
        config_filename = step_info["config_file"]

        # Skip if this step is not in the enabled steps list
        if enabled_steps is not None and step_name not in enabled_steps:
            continue

        # Build full path to config file
        config_path = config_dir / config_filename
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        resolved_steps.append((step_name, config_path))

    if not resolved_steps:
        if enabled_steps:
            print(f"No matching steps found for: {', '.join(enabled_steps)}")
        else:
            print(f"No steps found in {steps_yaml_path}")
        return

    total_steps = len(resolved_steps)
    stage_names = [name for name, _ in resolved_steps]

    from tt_boltz.boltzgen.progress import (
        DebugDisplay,
        RichDisplay,
        SilentDisplay,
        set_display,
        suppress_output,
    )

    debug = getattr(args, "debug", False)
    log = getattr(args, "log", False)
    if debug:
        display = DebugDisplay(stages=stage_names) if log else SilentDisplay()
    else:
        display = RichDisplay(stages=stage_names)
    set_display(display)

    # In Rich mode we hide task stdout/stderr so the live display stays clean.
    # In debug mode everything passes through unchanged.
    with display, suppress_output(active=not debug):
        for index, (step_name, config_path) in enumerate(resolved_steps, start=1):
            display.on_stage_start(step_name, index, total_steps)
            start = time.time()
            ok = True
            try:
                config = _load_yaml(config_path)
                config = _resolve_interpolations(config)
                task = _instantiate(config)
                if not isinstance(task, Task):
                    raise TypeError("Config must be an instance of Task.")
                task.run(config)
            except Exception:
                ok = False
                raise
            finally:
                display.on_stage_done(step_name, time.time() - start, ok=ok)


#### Pipeline implementation ####
@dataclass
class PipelineStep:
    name: str
    config_path: str
    args: List[str]

    def check(self):
        if self.name not in step_names:
            raise ValueError(
                f"Invalid step name: {self.name}. Available steps: {step_names}"
            )
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    def get_config(self) -> dict:
        config = _load_yaml(self.config_path)
        if self.args:
            config = _deep_merge(config, _dotlist_to_dict(self.args))
        return config


class BinderDesignPipeline:
    """
    The class itself does **not** execute work or write files; instead it builds a
    structured list of `PipelineStep` objects (name + config path + dot-list args)
    corresponding to the selected protocol and user flags. Those steps are then:

    1) **Written to yaml files** when `configure_command(...)` is executed
       - For each `PipelineStep`, the resolved config is written to
         `OUTPUT/config/<step>.yaml`.
       - A manifest `OUTPUT/steps.yaml` is also written, listing the enabled
         steps and their config files in execution order.

    2) **Executed from YAML** when `execute_command(...)` is executed —
       each step's config is instantiated in-process and ``Task.run(...)``
       is called directly. Subprocess mode existed in upstream BoltzGen
       for multi-GPU scheduling; on Tenstorrent we always run in-process
       so the device handle and program cache survive across steps.

    Pipeline task implementations live alongside this file:
        - Predict — tt_boltz/boltzgen/task/predict/predict.py
        - Analyze — tt_boltz/boltzgen/task/analyze/analyze.py
        - Filter  — tt_boltz/boltzgen/task/filter/filter.py

    Usage
    -----
    The class is constructed inside `configure_command(...)` and `run_command(...)`,
    not usually called directly by end users.

    Attributes
    ----------
    steps : list[PipelineStep]
        The ordered list of configured steps. Each entry carries the
        `name`, a `config_path` (base template under `--config_dir`), and a
        list of Hydra **dot-list** overrides that will be merged and written to YAML
        by `configure_command(...)`.

    Methods
    -------
    Notes
    -----
    * The pipeline distinguishes *where* configurations are **written** (in
      `configure_command`) from **how** they are **executed** (in `execute_command`).
      This separation makes it easy to tweak or re-run any step by editing the
      generated YAMLs.
    """

    def __init__(self, args: argparse.Namespace, moldir: Path):
        protocol = args.protocol
        if protocol not in protocol_configs:
            raise ValueError(
                f"Invalid protocol: {protocol}. Valid protocols: {list(protocol_configs.keys())}"
            )

        # Tenstorrent runs the heavy modules through ttnn; cuequivariance
        # kernels are GPU-only and would never fire here. Force use_kernels=False
        # regardless of the --use_kernels CLI flag.
        use_kernels = False
        print(f"Using kernels: {use_kernels} (Tenstorrent backend)")

        protocol_config = protocol_configs[protocol]
        print(f"Config overrides for protocol {protocol}: {protocol_config}")

        # Set protocol-specific and user-specified step specific args
        config_args_by_step = parse_config_args(
            protocol_config, args.config, step_names
        )

        # Tenstorrent runs as a single logical device per pipeline-step process.
        # Multi-card parallelism happens at the protein level via tt-boltz's
        # scheduler, not via Lightning DDP within a step.
        devices = args.devices if args.devices is not None else 1
        print(f"Using {devices} devices")

        # ``--steps`` restricts which pipeline stages run; checking enabled
        # *before* building each stage's config means we never resolve (and
        # so never download) artifacts for stages we won't execute.
        enabled = set(args.steps) if args.steps else None

        def _enabled(name: str) -> bool:
            return enabled is None or name in enabled

        self.steps = []

        # Design generation
        output_dir = args.output / "intermediate_designs"
        print(f"Raw designs will be saved to: {output_dir}")
        diffusion_batch_size = args.diffusion_batch_size
        if diffusion_batch_size is None:
            diffusion_batch_size = 1 if args.num_designs < 100 else 10
        num_batches = math.ceil(args.num_designs / diffusion_batch_size)
        print(f"Using diffusion batch size: {diffusion_batch_size}")
        print(f"Number of diffusion batches: {num_batches}")

        # Configure design checkpoint arguments. Only resolve (and download)
        # the design checkpoints if the design step will actually run.
        checkpoint_args = []
        if not args.only_inverse_fold and _enabled("design"):
            first_checkpoint_path = get_artifact_path(args, args.design_checkpoints[0])
            checkpoint_args.append(f"checkpoint={first_checkpoint_path}")
            if len(args.design_checkpoints) > 1:
                fraction_per_checkpoint = 1.0 / len(args.design_checkpoints)
                checkpoint_args.append(
                    f"override.checkpoints.first_checkpoint_num_samples={fraction_per_checkpoint}"
                )
                checkpoint_args.append(
                    f"override.checkpoints.checkpoint_list=["
                    + ",".join(
                        f"{{'checkpoint': {{'num_samples': {fraction_per_checkpoint}, 'path': '{get_artifact_path(args, checkpoint)}'}}}}"
                        for checkpoint in args.design_checkpoints[1:]
                    )
                    + "]"
                )
        design_step_and_noise_scale_args = []
        if args.step_scale is not None:
            design_step_and_noise_scale_args.append(
                f"override.diffusion_process_args.step_scale={args.step_scale}"
            )
            # Also disable the schedule when applying a fixed step scale
            design_step_and_noise_scale_args.append(
                f"override.step_scale_schedule=null"
            )
        if args.noise_scale is not None:
            design_step_and_noise_scale_args.append(
                f"override.diffusion_process_args.noise_scale={args.noise_scale}"
            )
            design_step_and_noise_scale_args.append(
                f"override.noise_scale_schedule=null"
            )

        if args.only_inverse_fold:
            exclude_residues = []
            inverse_fold_avoid = (
                args.inverse_fold_avoid
                if args.inverse_fold_avoid is not None
                else (
                    "C"
                    if protocol
                    in [
                        "peptide-anything",
                        "nanobody-anything",
                        "antibody-anything",
                    ]
                    else ""
                )
            )

            for one_letter_code in inverse_fold_avoid:
                exclude_residues.append(const.prot_letter_to_token[one_letter_code])

            if len(exclude_residues) > 0:
                print(
                    f"Inverse fold will avoid the following residues: {exclude_residues}"
                )
            print(f"Inverse-folded designs will be saved to: {output_dir}")
            # Designs from inverse folding
            if _enabled("inverse_folding"):
                self.steps.append(
                    PipelineStep(
                        name="inverse_folding",
                        config_path=args.config_dir / "inverse_fold_only.yaml",
                        args=[
                            f"output={output_dir}",
                            f"data.cfg.yaml_path=[{', '.join(str(s) for s in args.design_spec)}]",
                            f"trainer.devices={devices}",
                            f"data.cfg.multiplicity={getattr(args, 'inverse_fold_num_sequences', 10)}",
                            f"data.cfg.skip_existing={args.reuse}",
                            f"data.cfg.output_dir={output_dir}",
                            f"override.use_kernels={use_kernels}",
                            f"checkpoint={get_artifact_path(args, args.inverse_fold_checkpoint)}",
                            f"data.cfg.moldir={moldir}",
                            f"override.inverse_fold_args.inverse_fold_restriction=[{', '.join(exclude_residues)}]",
                        ]
                        + config_args_by_step.get("inverse_folding", []),
                    )
                )
        else:
            # Designs from diffusion model
            if _enabled("design"):
                self.steps.append(
                    PipelineStep(
                        name="design",
                        config_path=args.config_dir / "design.yaml",
                        args=[
                            f"output={output_dir}",
                            f"data.cfg.yaml_path=[{', '.join(str(s) for s in args.design_spec)}]",
                            f"trainer.devices={devices}",
                            f"data.num_workers={args.num_workers}",
                            f"data.cfg.skip_existing={args.reuse}",
                            f"data.cfg.multiplicity={num_batches}",
                            f"diffusion_samples={diffusion_batch_size}",
                            f"override.use_kernels={use_kernels}",
                            f"data.cfg.moldir={moldir}",
                        ]
                        + design_step_and_noise_scale_args
                        + checkpoint_args
                        + config_args_by_step["design"],
                    )
                )

            # Inverse folding of diffusion-generated backbones.
            if not args.skip_inverse_folding:
                exclude_residues = []
                inverse_fold_avoid = (
                    args.inverse_fold_avoid
                    if args.inverse_fold_avoid is not None
                    else (
                        "C"
                        if protocol in ["peptide-anything", "nanobody-anything", "antibody-anything"]
                        else ""
                    )
                )

                for one_letter_code in inverse_fold_avoid:
                    exclude_residues.append(const.prot_letter_to_token[one_letter_code])

                if len(exclude_residues) > 0:
                    print(
                        f"Inverse fold will avoid the following residues: {exclude_residues}"
                    )

                input_dir = output_dir
                output_dir = args.output / "intermediate_designs_inverse_folded"
                print(f"Inverse-folded designs will be saved to: {output_dir}")
                if _enabled("inverse_folding"):
                    self.steps.append(
                        PipelineStep(
                            name="inverse_folding",
                            config_path=args.config_dir / "inverse_fold.yaml",
                            args=[
                                f"output={output_dir}",
                                f"data.design_dir={input_dir}",
                                f"data.cfg.multiplicity={args.inverse_fold_num_sequences}",
                                f"data.cfg.num_workers={args.num_workers}",
                                f"data.skip_existing={args.reuse}",
                                f"data.skip_existing_kind=inverse_fold",
                                f"override.use_kernels={use_kernels}",
                                f"checkpoint={get_artifact_path(args, args.inverse_fold_checkpoint)}",
                                f"data.cfg.moldir={moldir}",
                                f"trainer.devices={devices}",
                                f"override.inverse_fold_args.inverse_fold_restriction=[{', '.join(exclude_residues)}]",
                            ]
                            + config_args_by_step["inverse_folding"],
                        )
                    )

        # Folding
        input_dir = output_dir
        if _enabled("folding"):
            self.steps.append(
                PipelineStep(
                    name="folding",
                    config_path=args.config_dir / "fold.yaml",
                    args=[
                        f"output={output_dir}",
                        f"data.design_dir={input_dir}",
                        f"trainer.devices={devices}",
                        f"data.cfg.num_workers={args.num_workers}",
                        f"data.skip_existing={args.reuse}",
                        f"data.skip_existing_kind=folded",
                        f"override.use_kernels={use_kernels}",
                        f"checkpoint={get_artifact_path(args, args.folding_checkpoint)}",
                        f"data.cfg.moldir={moldir}",
                    ]
                    + config_args_by_step["folding"],
                )
            )

        # Design folding
        input_dir = output_dir
        do_design_folding = protocol in ["protein-anything", "protein-small_molecule"]
        if do_design_folding and _enabled("design_folding"):
            self.steps.append(
                PipelineStep(
                    name="design_folding",
                    config_path=args.config_dir / "fold.yaml",
                    args=[
                        f"output={output_dir}",
                        f"data.design_dir={input_dir}",
                        f"trainer.devices={devices}",
                        f"data.cfg.num_workers={args.num_workers}",
                        f"data.skip_existing={args.reuse}",
                        f"data.skip_existing_kind=design_folded",
                        f"override.use_kernels={use_kernels}",
                        f"checkpoint={get_artifact_path(args, args.folding_checkpoint)}",
                        f"data.cfg.moldir={moldir}",
                        f"writer.designfolding=True",
                        f"data.cfg.return_designfolding=True",
                    ]
                    + config_args_by_step["design_folding"],
                )
            )

        # Affinity
        use_affinity = protocol in ["protein-small_molecule"]
        if use_affinity and _enabled("affinity"):
            self.steps.append(
                PipelineStep(
                    name="affinity",
                    config_path=args.config_dir / "affinity.yaml",
                    args=[
                        f"output={output_dir}",
                        f"data.design_dir={input_dir}",
                        f"trainer.devices={devices}",
                        f"data.cfg.num_workers={args.num_workers}",
                        f"data.skip_existing={args.reuse}",
                        f"data.skip_existing_kind=affinity",
                        f"override.use_kernels={use_kernels}",
                        f"checkpoint={get_artifact_path(args, args.affinity_checkpoint)}",
                        f"data.cfg.moldir={moldir}",
                    ]
                    + config_args_by_step["affinity"],
                )
            )

        # Analysis
        if _enabled("analysis"):
            self.steps.append(
                PipelineStep(
                    name="analysis",
                    config_path=args.config_dir / "analysis.yaml",
                    args=[
                        f"design_dir={input_dir}",
                        f"data.skip_existing={args.reuse}",
                        f"data.skip_existing_kind=analyzed",
                        f"data.cfg.moldir={moldir}",
                        f"designfolding_metrics={do_design_folding}",
                        f"delta_sasa_original={args.skip_inverse_folding}",
                        f"noncovalents_original={args.skip_inverse_folding}",
                        f"allatom_fold_metrics={args.skip_inverse_folding}",
                    ]
                    + config_args_by_step["analysis"],
                )
            )

        # Filtering
        output_dir = args.output / "final_ranked_designs"
        print(f"Final ranked designs will be saved to: {output_dir}")

        # Build filter arguments
        filter_args = [
            f"design_dir={input_dir}",
            f"outdir={args.output}",
            f"from_inverse_folded={not args.skip_inverse_folding}",
            f"use_affinity={use_affinity}",
            f"filter_designfolding={do_design_folding}",
            f"budget={args.budget}",
        ]

        # Add optional filtering arguments
        if args.alpha is not None:
            filter_args.append(f"alpha={args.alpha}")
        if args.filter_biased is not None:
            filter_args.append(f"filter_biased={args.filter_biased}")
        if args.refolding_rmsd_threshold is not None:
            filter_args.append(
                f"refolding_rmsd_threshold={args.refolding_rmsd_threshold}"
            )
        if args.metrics_override is not None:
            parsed_metrics = parse_metrics_override(args.metrics_override)
            print(f"Filtering metrics override: {parsed_metrics}")
            filter_args.append(f"metrics_override={parsed_metrics}")
        if args.additional_filters is not None:
            parsed_filters = parse_additional_filters(args.additional_filters)
            print(f"Filtering additional filters: {parsed_filters}")
            filter_args.append(f"additional_filters={parsed_filters}")
        if args.size_buckets is not None:
            parsed_size_buckets = parse_size_buckets(args.size_buckets)
            print(f"Filtering size buckets: {parsed_size_buckets}")
            filter_args.append(f"size_buckets={parsed_size_buckets}")

        if _enabled("filtering"):
            self.steps.append(
                PipelineStep(
                    name="filtering",
                    config_path=args.config_dir / "filtering.yaml",
                    args=filter_args + config_args_by_step["filtering"],
                )
            )

    def pretty_print(self):
        for i, step in enumerate(self.steps):
            print(f"[{i + 1}] {step.name:25s}")


### Misc utiltiies ###
def check_design_specs(args: argparse.Namespace, moldir: Path, mols: Dict[str, Any]):
    last_banner = ""
    for design_spec in args.design_spec:
        banner = f"************** Checking design spec: {design_spec} **************"
        last_banner = banner
        print(banner)
        check_design_spec(args, moldir, design_spec, mols)
    if last_banner:
        print("*" * len(last_banner))


def check_design_spec(
    args: argparse.Namespace, moldir: Path, design_spec: Path, mols: Dict[str, Any]
):
    """
    Validate a single design specification, color/annotate its structure, and
    write an mmCIF visualization.

    This function parses a YAML design spec with `YamlDesignParser`, applies visual
    annotations to the parsed structure (via B-factors and per-residue color features),
    reports unresolved residues/atoms, and writes a colored mmCIF file if `--output`
    was provided on the CLI.

    - Writes `<output>/<design_spec.stem>.cif` if `args.output` is set.

        * B-factor encodes design/binding (100 for designed, +80 if binding).
        * `design_color_features`: 1.0 for binding residues, 0.8 otherwise.

    Examples
    --------
    From the CLI (via the `check` subcommand):

        $ boltzgen check path/to/design.yaml --output checked/

    From Python:

        moldir = PATH TO MOLDIR
        mols = load_canonicals(moldir=moldir)
        check_design_spec(args, moldir, Path("design.yaml"), mols)
    """
    parser = YamlDesignParser(moldir)
    parsed = parser.parse_yaml(design_spec, mols, moldir)
    structure = parsed.structure
    design_info = parsed.design_info
    design_color_features = np.ones_like(design_info.res_binding_type) * 0.8
    design_color_features[design_info.res_binding_type.astype(bool)] = 1.0
    extract_mask = np.zeros(len(structure.residues), dtype=bool)
    for i, residue in enumerate(structure.residues):
        structure.atoms["bfactor"][
            residue["atom_idx"] : residue["atom_idx"] + residue["atom_num"]
        ] = 100 * design_info.res_design_mask[i] + 80 * design_info.res_binding_type[i]

        atom_positions = structure.atoms["coords"][
            residue["atom_idx"] : residue["atom_idx"] + residue["atom_num"]
        ]
        zero_position_count = ((atom_positions**2).sum(axis=1) < 1e-6).sum()
        if not zero_position_count == len(atom_positions):
            extract_mask[i] = True

    structure_write = structure
    if extract_mask.sum() > 0:
        structure_write = structure.extract_residues(structure, extract_mask)
        design_color_features = design_color_features[extract_mask]

    mmcif = to_mmcif(
        structure_write,
        design_coloring=True,
        color_features=design_color_features,
    )

    # Check for unresolved residues and atoms to log a warning
    unresolved_residues = (~structure.residues["is_present"]).nonzero()[0]
    unresolved_atoms = (~structure.atoms["is_present"]).nonzero()[0]

    print(f"Total designed residues: {design_info.res_design_mask.sum()}")

    if len(unresolved_residues) > 0 or len(unresolved_atoms) > 0:
        atom_to_residue = {}
        for residue_idx, residue in enumerate(structure.residues):
            start = residue["atom_idx"]
            end = start + residue["atom_num"]
            for i in range(start, end):
                atom_to_residue[i] = residue_idx

        residue_to_chain = {}
        residue_in_chain = {}
        for chain in structure.chains:
            start = chain["res_idx"]
            end = start + chain["res_num"]
            j = 1
            for i in range(start, end):
                residue_to_chain[i] = chain["name"]
                residue_in_chain[i] = j
                j += 1

        atom_to_chain = {}
        for chain in structure.chains:
            start = chain["atom_idx"]
            end = start + chain["atom_num"]
            for i in range(start, end):
                atom_to_chain[i] = chain["name"]

        msg = f"There are {len(unresolved_residues)} unresolved residues and {len(unresolved_atoms)} unresolved atoms in the target."
        print(msg)

    if args.output is not None:
        output_path = args.output / (design_spec.stem + ".cif")
    else:
        output_path = design_spec.stem + ".cif"
    with open(output_path, "w") as f:
        f.write(mmcif)
    print(f"Design specification visualization is written to {str(output_path)}")


def get_artifact_path(
    args, artifact: str, repo_type: str = "model", verbose: bool = True
) -> Path:
    """Resolve an artifact spec to a local file path.

    Two accepted forms:
      * ``http(s)://...``  — fetched once via ``urllib`` into the local cache
        (default ``~/.boltz/boltzgen/``; override with ``--cache``). Used by
        the GCP-hosted defaults in ``ARTIFACTS``.
      * Anything else — treated as a local file path.
    """
    if artifact.startswith(("http://", "https://")):
        import urllib.request

        cache = args.cache if args.cache is not None else (Path.home() / ".boltz" / "boltzgen")
        cache.mkdir(parents=True, exist_ok=True)
        result = cache / artifact.rsplit("/", 1)[-1]
        if args.force_download or not result.exists():
            print(f"Downloading {artifact} → {result}")
            urllib.request.urlretrieve(artifact, result)
    else:
        result = Path(artifact)
    if not result.exists():
        raise FileNotFoundError(f"Model not found: {result}")
    if verbose:
        print(f"Using {repo_type} artifact: {result}")
    return result


def parse_config_args(base_config, config_args, valid_step_names):
    config_args_by_step = collections.defaultdict(list)
    config_args_by_step.update(base_config)
    if config_args:
        for config in config_args:
            if len(config) < 2:
                raise ValueError(
                    f"Invalid config: {config}. Expected format: <step_name> <arg1>=<value1> <arg2>=<value2> ..."
                )
            step_name = config[0]
            if step_name not in valid_step_names:
                raise ValueError(
                    f"Invalid step name: {step_name}. Available steps: {valid_step_names}"
                )
            key_value_pairs = config[1:]
            config_args_by_step[step_name].extend(key_value_pairs)
    return config_args_by_step


### Filtering argument parsing functions ####
def parse_metrics_override(value_list):
    """Parse metrics_override from key=value pairs."""
    if not value_list:
        return None

    metrics_override = {}
    for item in value_list:
        if "=" in item:
            key, value = item.split("=", 1)
            if value == "" or value.lower() == "none":
                metrics_override[key] = None  # Remove metric
            else:
                try:
                    metrics_override[key] = float(value)
                except ValueError:
                    raise ValueError(
                        f"Invalid weight value for metric '{key}': '{value}'. Must be a number."
                    )
        else:
            raise ValueError(
                f"Invalid metrics_override format: '{item}'. Use 'metric_name=weight' format."
            )
    return metrics_override


def parse_additional_filters(value_list):
    """Parse additional_filters from feature>threshold or feature<threshold format."""
    if not value_list:
        return None

    additional_filters = []
    for item in value_list:
        if ">" in item:
            feature, threshold_str = item.split(">", 1)
            try:
                threshold = float(threshold_str)
                additional_filters.append(
                    {
                        "feature": feature,
                        "threshold": threshold,
                        "lower_is_better": False,  # > means higher is better
                    }
                )
            except ValueError:
                raise ValueError(
                    f"Invalid threshold value: '{threshold_str}'. Must be a number."
                )
        elif "<" in item:
            feature, threshold_str = item.split("<", 1)
            try:
                threshold = float(threshold_str)
                additional_filters.append(
                    {
                        "feature": feature,
                        "threshold": threshold,
                        "lower_is_better": True,  # < means lower is better
                    }
                )
            except ValueError:
                raise ValueError(
                    f"Invalid threshold value: '{threshold_str}'. Must be a number."
                )
        else:
            raise ValueError(
                f"Invalid additional_filters format: '{item}'. Use 'feature>threshold' or 'feature<threshold' format."
            )
    return additional_filters


def parse_size_buckets(value_list):
    """Parse size_buckets from min-max:count format."""
    if not value_list:
        return None

    size_buckets = []
    for item in value_list:
        if ":" in item and "-" in item:
            range_part, count_str = item.split(":", 1)
            if "-" in range_part:
                min_str, max_str = range_part.split("-", 1)
                try:
                    min_size = int(min_str)
                    max_size = int(max_str)
                    count = int(count_str)
                    size_buckets.append(
                        {"num_designs": count, "min": min_size, "max": max_size}
                    )
                except ValueError as e:
                    if "invalid literal" in str(e):
                        raise ValueError(
                            f"Invalid size_buckets format: '{item}'. All values must be integers. Use 'min-max:count' format."
                        )
                    else:
                        raise e
            else:
                raise ValueError(
                    f"Invalid size_buckets format: '{item}'. Use 'min-max:count' format."
                )
        else:
            raise ValueError(
                f"Invalid size_buckets format: '{item}'. Use 'min-max:count' format."
            )
    return size_buckets


def merge_command(args: argparse.Namespace) -> None:
    """
    Merge multiple BoltzGen output directories into a single destination directory so
    the filtering step can be rerun over the combined set of designs.
    """

    def _merge_design_dir(
        sources: list[Path],
        run_tags: dict[Path, str],
        dir_name: str,
        dest_dir: Path,
        id_map: dict[tuple[Path, str], str],
    ) -> int:
        metrics_frames: list[pd.DataFrame] = []
        seq_frames: list[pd.DataFrame] = []
        per_target_frames: list[pd.DataFrame] = []
        merged_count = 0

        for root in sources:
            src_dir = root / dir_name
            if not src_dir.exists():
                continue

            run_tag = run_tags[root]
            source_mappings: list[tuple[str, str, str, str]] = []

            metrics_path = src_dir / "aggregate_metrics_analyze.csv"
            if metrics_path.exists():
                df = pd.read_csv(metrics_path)
                if not df.empty:
                    updated_rows = []
                    for _, row in df.iterrows():
                        if "id" not in row or "file_name" not in row:
                            raise ValueError(
                                "aggregate_metrics_analyze.csv must contain 'id' and 'file_name' columns."
                            )
                        original_id = str(row["id"])
                        original_file = str(row["file_name"])
                        key = (root, original_id)
                        new_id = id_map.setdefault(key, f"{run_tag}_{original_id}")
                        new_file = _make_new_file_name(original_file, new_id)
                        updated_rows.append(
                            {**row, "id": new_id, "file_name": new_file}
                        )
                        source_mappings.append(
                            (original_id, new_id, original_file, new_file)
                        )
                    metrics_frames.append(pd.DataFrame(updated_rows))
                    merged_count += len(source_mappings)
            else:
                known_ids = [
                    (orig, new_id)
                    for (src, orig), new_id in id_map.items()
                    if src == root
                ]
                for original_id, new_id in known_ids:
                    original_file = f"{original_id}.cif"
                    new_file = _make_new_file_name(original_file, new_id)
                    source_mappings.append(
                        (original_id, new_id, original_file, new_file)
                    )

            if not source_mappings:
                continue

            dest_dir.mkdir(parents=True, exist_ok=True)

            seq_path = src_dir / "ca_coords_sequences.pkl.gz"
            if seq_path.exists():
                seq_df = pd.read_pickle(seq_path)
                original_ids = [orig for orig, _, _, _ in source_mappings]
                seq_subset = seq_df[seq_df["id"].astype(str).isin(original_ids)].copy()
                if not seq_subset.empty:
                    id_lookup = {orig: new for orig, new, _, _ in source_mappings}
                    seq_subset["id"] = seq_subset["id"].astype(str).map(id_lookup)
                    seq_frames.append(seq_subset)

            per_target_path = src_dir / "per_target_metrics_analyze.csv"
            if per_target_path.exists():
                per_target_frames.append(pd.read_csv(per_target_path))

            for original_id, new_id, original_file, new_file in source_mappings:
                _copy_design_files(
                    src_dir=src_dir,
                    dest_dir=dest_dir,
                    original_id=original_id,
                    new_id=new_id,
                    original_file=original_file,
                    new_file=new_file,
                    include_refold=True,
                )

        if metrics_frames:
            pd.concat(metrics_frames, ignore_index=True).to_csv(
                dest_dir / "aggregate_metrics_analyze.csv", index=False
            )
        if seq_frames:
            pd.concat(seq_frames, ignore_index=True).to_pickle(
                dest_dir / "ca_coords_sequences.pkl.gz", compression="gzip"
            )
        if per_target_frames:
            pd.concat(per_target_frames, ignore_index=True).to_csv(
                dest_dir / "per_target_metrics_analyze.csv", index=False
            )

        return merged_count


    def _copy_design_files(
        src_dir: Path,
        dest_dir: Path,
        original_id: str,
        new_id: str,
        original_file: str,
        new_file: str,
        include_refold: bool,
    ) -> None:
        _copy_path(src_dir / original_file, dest_dir / new_file, required=True)
        _copy_path(
            src_dir / f"{original_id}.npz",
            dest_dir / f"{new_id}.npz",
            required=False,
        )
        _copy_path(
            src_dir / f"{original_id}_native.cif",
            dest_dir / f"{new_id}_native.cif",
            required=False,
        )
        _copy_path(
            src_dir / f"{original_id}_native.pdb",
            dest_dir / f"{new_id}_native.pdb",
            required=False,
        )
        if include_refold:
            _copy_path(
                src_dir / const.refold_cif_dirname / original_file,
                dest_dir / const.refold_cif_dirname / new_file,
                required=False,
            )
            _copy_path(
                src_dir / const.refold_design_cif_dirname / original_file,
                dest_dir / const.refold_design_cif_dirname / new_file,
                required=False,
            )


    def _make_new_file_name(original_file: str, new_id: str) -> str:
        path = Path(original_file)
        suffix = "".join(path.suffixes)
        return f"{new_id}{suffix}" if suffix else new_id


    def _slugify_run_tag(path: Path, index: int) -> str:
        slug = re.sub(r"[^0-9A-Za-z]+", "-", path.name).strip("-").lower()
        return slug or f"run{index}"


    def _copy_path(src: Path, dst: Path, *, required: bool) -> None:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            # Try to make a hard link if possible, otherwise copy
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        elif required:
            raise FileNotFoundError(f"Required file missing during merge: {src}")

    if not args.sources:
        raise ValueError("Provide at least one source directory to merge.")

    dest_root = args.output.expanduser().resolve()

    source_roots: list[Path] = []
    for src in args.sources:
        root = Path(src).expanduser().resolve()
        if root == dest_root:
            print(f"Skipping {root} as it is the destination directory")
            continue
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Source directory not found: {root}")
        source_roots.append(root)

    dest_root.mkdir(parents=True, exist_ok=True)

    run_tags = {
        root: _slugify_run_tag(root, idx + 1) for idx, root in enumerate(source_roots)
    }
    id_map: dict[tuple[Path, str], str] = {}

    total_designs = 0
    for dir_name in [
        "intermediate_designs_inverse_folded",
        "intermediate_designs",
    ]:
        dest_dir = dest_root / dir_name
        merged = _merge_design_dir(
            sources=source_roots,
            run_tags=run_tags,
            dir_name=dir_name,
            dest_dir=dest_dir,
            id_map=id_map,
        )
        if merged:
            total_designs += merged
            print(f"- merged {merged} designs into {dest_dir}")

    if total_designs == 0:
        print("No designs found to merge.")
    else:
        print("===============================================")
        print(f"Merged {len(source_roots)} source(s) into {dest_root}")
        print(f"Total designs available for filtering: {total_designs}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_command(args)
    elif args.command == "configure":
        configure_command(args)
    elif args.command == "execute":
        execute_command(args)
    elif args.command == "download":
        download_command(args)
    elif args.command == "check":
        check_command(args)
    elif args.command == "merge":
        merge_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
