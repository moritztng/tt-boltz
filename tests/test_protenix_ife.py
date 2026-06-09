"""On-device parity for the Protenix-v2 InputFeatureEmbedder atom encoder -> s_inputs.

Validates tt_bio.protenix.AtomAttentionEncoder (featurization + p_lm augmentation +
windowed atom transformer + token aggregation + concat) vs golden s_inputs from the
real v2 reference (scripts/protenix_extract_ife.py -> ~/protenix_ife_gold.pkl).
"""
import os, pickle, pytest, torch, ttnn

_GOLD = os.path.expanduser("~/protenix_ife_gold.pkl")
pytestmark = pytest.mark.skipif(not os.path.exists(_GOLD), reason="golden pkl missing; run scripts/protenix_extract_ife.py")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def test_input_feature_embedder_on_device():
    from tt_bio.tenstorrent import get_device
    from tt_bio.protenix import AtomAttentionEncoder
    g = pickle.load(open(_GOLD, "rb"))
    W, F = g["aae_state"], g["feat"]
    N = F["ref_pos"].shape[0]
    dev = get_device()
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    def tt(x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    enc = AtomAttentionEncoder(W, ck)
    f_in = torch.cat([F["ref_mask"].reshape(N, 1), F["ref_element"].reshape(N, 128),
                      F["ref_atom_name_chars"].reshape(N, 256)], dim=-1)
    nb, nq, nk, _ = F["d_lm"].shape; M = nb * nq * nk
    d = F["d_lm"].reshape(M, 3); v = F["v_lm"].reshape(M, 1)
    invd = (1.0 / (1.0 + (F["d_lm"] ** 2).sum(-1, keepdim=True))).reshape(M, 1)
    mt = g["mask_trunked"].reshape(M, 1).float()
    a2t = F["atom_to_token_idx"].long(); NT = int(a2t.max()) + 1
    Mmat = torch.zeros(NT, N)
    for a in range(N):
        Mmat[a2t[a], a] = 1.0
    Mmat = Mmat / (Mmat.sum(-1, keepdim=True) + 1e-6)
    dm = F["deletion_mean"]; dm = dm.reshape(-1, 1) if dm.dim() == 1 else dm
    out = enc(tt(F["ref_pos"]), tt(torch.arcsinh(F["ref_charge"]).reshape(N, 1)),
              tt(F["ref_mask"].reshape(N, 1)), tt(f_in), tt(d), tt(v), tt(invd), mt, tt(Mmat),
              tt(F["restype"]), tt(F["profile"]), tt(dm))
    out = ttnn.to_torch(out).float()[:NT]
    assert _pcc(out, g["golden_sinputs"].float()) > 0.999
