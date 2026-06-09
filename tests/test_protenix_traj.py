# On-device end-to-end diffusion test: replay the reference v2 sampler trajectory and
# assert the on-device denoiser matches the reference denoised coords at EVERY sampling
# step (full sigma range). Gated on the golden pkls (built in the py3.11 reference venv
# via scripts/protenix_extract_traj.py). Runs the replay script in the system python3
# (ttnn) as a subprocess and checks the reported all-step PCC floor.
import os, re, sys, subprocess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ = os.path.expanduser("~/protenix_traj.pkl")
PRE = os.path.expanduser("~/protenix_denoiser_pre.pkl")
CKPT = os.path.expanduser("~/protenix_ckpt/protenix-v2.pt")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(TRAJ) and os.path.exists(PRE) and os.path.exists(CKPT)),
    reason="needs ~/protenix_traj.pkl, ~/protenix_denoiser_pre.pkl, v2 ckpt (build via scripts/protenix_extract_traj.py)",
)


def test_protenix_diffusion_trajectory_on_device():
    """Per-step on-device denoiser PCC vs reference >= 0.999 across the full schedule."""
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "protenix_traj_replay.py")],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    ).stdout
    m = re.search(r"ALL-STEP denoiser PCC: min ([0-9.]+)\s+mean ([0-9.]+)", out)
    assert m is not None, f"replay did not report PCC summary:\n{out[-2000:]}"
    pcc_min, pcc_mean = float(m.group(1)), float(m.group(2))
    assert pcc_min >= 0.999, f"min per-step denoiser PCC {pcc_min} < 0.999\n{out[-2000:]}"
    assert pcc_mean >= 0.9995, f"mean per-step denoiser PCC {pcc_mean} < 0.9995"
