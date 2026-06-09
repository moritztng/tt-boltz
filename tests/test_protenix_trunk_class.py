# On-device test: the production tt_bio.protenix.Trunk class reproduces the full
# 10-cycle v2 trunk vs the real reference (s/z PCC). Gated on the golden pkls. Runs the
# trunk-class validation script in system python3 (ttnn) and checks the PCC floor.
import os, re, sys, subprocess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEED = [os.path.expanduser(p) for p in (
    "~/protenix_ref_out.pkl", "~/protenix_trunkin_gold.pkl", "~/protenix_trunk_gold.pkl",
    "~/protenix_ckpt/protenix-v2.pt")]

pytestmark = pytest.mark.skipif(not all(os.path.exists(p) for p in NEED),
                                reason="needs v2 trunk golden pkls + ckpt")


def test_protenix_trunk_class_on_device():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "protenix_trunk_class.py")],
                         cwd=ROOT, capture_output=True, text=True, timeout=900).stdout
    m = re.search(r"TRUNK CLASS \(10 cycles\)\s+s PCC ([0-9.]+)\s+z PCC ([0-9.]+)", out)
    assert m is not None, f"no PCC summary:\n{out[-2000:]}"
    s_pcc, z_pcc = float(m.group(1)), float(m.group(2))
    assert s_pcc >= 0.98, f"trunk s PCC {s_pcc} < 0.98\n{out[-1500:]}"
    assert z_pcc >= 0.98, f"trunk z PCC {z_pcc} < 0.98"
