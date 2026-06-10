# Large-N regression guard: a 512-residue fold exercises the pair-tensor chunking paths
# that previously OOM'd (diffusion-conditioning Transition on (N,N,256), and the Transition
# 4D row-chunk for c_z=256). Asserts the fold completes with finite, non-collapsed coords.
# Gated on the v2 checkpoint; slow (~2 min, trunk-dominated) so marked accordingly.
import os
import sys

import pytest

CKPT = os.environ.get("PROTENIX_CKPT", os.path.expanduser("~/protenix_ckpt/protenix-v2.pt"))
pytestmark = pytest.mark.skipif(not os.path.exists(CKPT), reason="needs the v2 checkpoint")

# deterministic pseudo-random 512-residue sequence (no external fixtures needed)
_AA = "ARNDCQEGHILKMFPSTWYV"
_SEQ512 = "".join(_AA[(i * 7 + 13) % 20] for i in range(512))


def test_fold_512_no_oom():
    import torch
    import ttnn

    from tt_bio.protenix import Protenix
    from tt_bio.protenix_data import build_protein_features
    from tt_bio.tenstorrent import get_device

    feats = build_protein_features(_SEQ512)  # single-sequence exercises the pair-tensor path
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    model = Protenix.load_from_checkpoint(CKPT, compute_kernel_config=ckc, device=dev)
    coords = model.fold(feats, n_step=2, n_sample=1, seed=0)  # 2 steps: we test memory, not accuracy
    assert coords.shape[0] == 1 and coords.shape[2] == 3
    assert torch.isfinite(coords).all(), "non-finite coords"
    rg = float((coords[0] - coords[0].mean(0)).pow(2).sum(-1).mean().sqrt())
    assert rg > 5.0, f"structure collapsed (Rg {rg})"
