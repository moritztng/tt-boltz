"""End-to-end numerical regression against a saved CUDA GPU reference.

Runs the full design pipeline on the same input the GPU reference was
generated from and compares backbone bond-length distributions of the
designed structure against the GPU baseline. Catches:

  * dead diffusion sampler (no movement → distorted geometry)
  * broken peptide bonds (covalent chemistry violated)
  * structure-writer regressions
  * any downstream-stage breakage that perturbs final coordinates

Bond *chemistry* is invariant to the BoltzGen architectural knobs that
differ between upstream BoltzGen and our vendored fork (``cyclic_pos_enc``,
``recycling_detach``, several validator/training fields), so this test
works even though a per-element trunk-activation comparison against the
GPU reference does *not* (the upstream constructor accepts ~15 extra args
that our slimmed Boltz drops, which perturbs trunk numerics out of bf16
tolerance while leaving bond chemistry intact).

The reference data is a 36 MB tarball at
``https://storage.googleapis.com/tt-boltz-artifacts/boltzgen_regression.tar.gz``,
downloaded once on first run to ``~/.cache/tt-boltz/regression/``. The
tarball's README documents how to regenerate it.
"""
from __future__ import annotations

import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict

import pytest


REGRESSION_URL = (
    "https://storage.googleapis.com/tt-boltz-artifacts/boltzgen_regression.tar.gz"
)
_CACHE = Path.home() / ".cache/tt-boltz/regression"
_ROOT = _CACHE / "tt_boltz_regression_v1"

_DESIGN_CKPT = (
    Path.home()
    / ".cache/huggingface/hub/models--boltzgen--boltzgen-1"
    / "snapshots/c1be29e1f82ffcc72264f64b993c43fb4e0d17f0/boltzgen1_diverse.ckpt"
)


def _fetch_regression_data() -> Path:
    """Download + extract the GPU reference tarball if not already cached."""
    if (_ROOT / "README.md").exists():
        return _ROOT
    _CACHE.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE / "regression.tar.gz"
    print(f"Fetching {REGRESSION_URL} ...")
    urllib.request.urlretrieve(REGRESSION_URL, tmp)
    with tarfile.open(tmp) as tar:
        tar.extractall(_CACHE)
    tmp.unlink()
    assert (_ROOT / "README.md").exists(), "tarball did not contain expected layout"
    return _ROOT


@pytest.fixture(scope="module")
def regression_data() -> Path:
    if not _DESIGN_CKPT.exists():
        pytest.skip(f"design checkpoint missing: {_DESIGN_CKPT}")
    try:
        return _fetch_regression_data()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not fetch regression tarball: {exc}")


def _bond_stats_from_cif(path: Path) -> Dict[str, Dict[str, float]]:
    """Per-bond-type mean and std (Å) for backbone bonds in a structure cif."""
    import gemmi
    import numpy as np

    st = gemmi.read_structure(str(path))
    entries = []  # (chain_name, N pos, CA pos, C pos)
    for model in st:
        for chain in model:
            for res in chain:
                atoms = {a.name: a.pos for a in res}
                if {"N", "CA", "C"} <= atoms.keys():
                    entries.append((chain.name, atoms["N"], atoms["CA"], atoms["C"]))

    def _dist(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5

    n_ca = np.array([_dist(n, ca) for _, n, ca, _ in entries])
    ca_c = np.array([_dist(ca, c) for _, _, ca, c in entries])
    c_n_next = np.array([
        _dist(entries[i][3], entries[i + 1][1])
        for i in range(len(entries) - 1)
        if entries[i][0] == entries[i + 1][0]
    ])
    return {
        "N-CA":  {"mean": float(n_ca.mean()),     "std": float(n_ca.std())},
        "CA-C":  {"mean": float(ca_c.mean()),     "std": float(ca_c.std())},
        "C-N+1": {"mean": float(c_n_next.mean()), "std": float(c_n_next.std())},
    }


def test_designed_structure_bond_geometry_matches_gpu(
    regression_data: Path, tmp_path: Path
) -> None:
    """End-to-end: full pipeline run; backbone bond means within 0.02 Å of GPU.

    Runs at ``sampling_steps=500`` (production default) to match the
    sampling level the GPU reference was captured at. Lower step counts
    leave the diffusion sampler under-converged — bond-length std balloons
    from ~0.01 Å (converged) to ~0.25 Å (20 steps), and the mean drift gets
    swamped by sampling noise rather than reflecting real numerical drift.

    Total runtime ~10 min on Blackhole (design step dominates). Set
    ``BOLTZGEN_REGRESSION_FAST=1`` to downshift to ``sampling_steps=20``
    for local sanity checks — the assertion thresholds widen automatically.

    Engh–Huber crystal-structure reference values (Å):
        N–Cα  1.459 ± 0.020
        Cα–C  1.525 ± 0.021
        C–N+1 1.329 ± 0.014

    We don't compare per-residue identity — the designed sequence varies
    stochastically with the diffusion seed — only the chemistry-defining
    bond geometry of whatever sequence emerged.
    """
    import os

    fast = bool(os.environ.get("BOLTZGEN_REGRESSION_FAST"))
    sampling_steps = 20 if fast else 500
    # At low step counts the diffusion sampler is far from converged so
    # bond-length distributions are wide — widen the tolerance accordingly.
    mean_drift_max = 0.05 if fast else 0.02
    # TT runs the diffusion sampler with a bf16 score model whereas the GPU
    # reference is fp32 — measured ratio on a clean run is ~3× across all
    # three bond types. 5× gives headroom; a catastrophic dispersion
    # (sampler collapse, exploded bonds) would push the ratio well past it.
    std_ratio_max  = 30.0 if fast else 5.0
    tt_boltz_bin = shutil.which("tt-boltz")
    if tt_boltz_bin is None:
        pytest.skip("tt-boltz CLI not on PATH; activate the venv before running")

    # Stage a working copy of the spec under its original filename:
    #  * the yaml's design "id" is derived from the yaml filename stem,
    #    and the GPU reference structure was captured as ``input.cif`` —
    #    naming the yaml ``input.yaml`` makes the output paths match.
    #  * the yaml references the target via the relative path ``1g13.cif``,
    #    but the tarball renamed it to ``target.cif`` for clarity.
    work = tmp_path / "spec"
    work.mkdir()
    spec_yaml = work / "input.yaml"
    spec_yaml.write_text((regression_data / "input/design_spec.yaml").read_text())
    shutil.copy(regression_data / "input/target.cif", work / "1g13.cif")

    out_dir = tmp_path / "regression_out"
    cmd = [
        tt_boltz_bin, "gen", "run", str(spec_yaml),
        "--output", str(out_dir),
        "--num_designs", "1",
        # Pin to one card: this asserts the single-device output layout
        # (intermediate_designs/<stem>.cif), and on a multi-card box the default
        # all-cards fan-out would split one design across workers.
        "--devices", "1",
        "--config", "design", f"sampling_steps={sampling_steps}",
    ]
    subprocess.check_call(cmd)

    produced = out_dir / "intermediate_designs" / f"{spec_yaml.stem}.cif"
    assert produced.exists(), f"pipeline produced no design at {produced}"

    tt_stats = _bond_stats_from_cif(produced)
    gpu_stats = _bond_stats_from_cif(regression_data / "reference/designed_structure.cif")
    print(f"TT  bond stats: {tt_stats}")
    print(f"GPU bond stats: {gpu_stats}")

    for bond in ("N-CA", "CA-C", "C-N+1"):
        mean_drift = abs(tt_stats[bond]["mean"] - gpu_stats[bond]["mean"])
        assert mean_drift < mean_drift_max, (
            f"{bond}: TT mean {tt_stats[bond]['mean']:.4f} drifts "
            f"{mean_drift:.4f} Å from GPU {gpu_stats[bond]['mean']:.4f} "
            f"(>{mean_drift_max})"
        )
        std_ratio = tt_stats[bond]["std"] / max(gpu_stats[bond]["std"], 1e-6)
        assert std_ratio < std_ratio_max, (
            f"{bond}: TT std {tt_stats[bond]['std']:.4f} is {std_ratio:.1f}× "
            f"GPU std {gpu_stats[bond]['std']:.4f} (>{std_ratio_max})"
        )
