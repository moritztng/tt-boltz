# Capstone on-device test: the full tt_bio.protenix.Protenix.fold pipeline (atom encoder
# -> diffusion atom cache -> 10-cycle trunk -> diffusion pair/single conditioning -> EDM
# sampler) runs end-to-end on real v2 weights and produces a valid (finite, non-collapsed)
# structure whose RMSD to the reference is within the sampler's own seed-to-seed variance.
# Gated on the golden feats pkls + v2 ckpt.
import os, re, sys, subprocess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEED = [os.path.expanduser(p) for p in (
    "~/protenix_ife_gold.pkl", "~/protenix_trunkin_gold.pkl", "~/protenix_ref_out.pkl",
    "~/protenix_traj.pkl", "~/protenix_ckpt/protenix-v2.pt")]

pytestmark = pytest.mark.skipif(not all(os.path.exists(p) for p in NEED),
                                reason="needs v2 golden feats pkls + ckpt")


def test_protenix_fold_end_to_end():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "protenix_fold_e2e.py")],
                         cwd=ROOT, capture_output=True, text=True, timeout=900).stdout
    assert "FOLD_E2E_DONE" in out, f"fold did not finish:\n{out[-2000:]}"
    assert "finite=True" in out, f"non-finite coords:\n{out[-2000:]}"
    rg = float(re.search(r"Rg ([0-9.]+) A", out).group(1))
    s2s = float(re.search(r"seed-to-seed Kabsch RMSD: ([0-9.]+)", out).group(1))
    vref = float(re.search(r"seed0 vs reference Kabsch RMSD: ([0-9.]+)", out).group(1))
    assert rg > 10.0, f"structure collapsed (Rg {rg} A < 10) -- conditioning bug\n{out[-1500:]}"
    # the on-device sample must lie within the sampler's own variance band of the reference
    assert vref <= s2s * 1.4 + 1.0, f"vs-reference RMSD {vref} A exceeds variance band (seed-to-seed {s2s} A)"
