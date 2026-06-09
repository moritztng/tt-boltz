"""On-device parity for the Protenix-v2 AtomTransformer (3-block windowed atom attn).

Validates tt_bio.protenix.AtomTransformer (fully on-device) against golden_qout from
the real v2 reference (scripts/protenix_extract_atomtx.py -> ~/protenix_atomtx_gold.pkl).
"""
import os, pickle, pytest, ttnn

_GOLD = os.path.expanduser("~/protenix_atomtx_gold.pkl")
pytestmark = pytest.mark.skipif(not os.path.exists(_GOLD), reason="golden pkl missing; run scripts/protenix_extract_atomtx.py")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def test_atom_transformer_on_device():
    from tt_bio.tenstorrent import get_device
    from tt_bio.protenix import AtomTransformer
    g = pickle.load(open(_GOLD, "rb"))
    dev = get_device()
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    def tt(x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    atx = AtomTransformer(3, g["weights"], ck)
    out = ttnn.to_torch(atx(tt(g["q"].unsqueeze(0)), tt(g["c"].unsqueeze(0)), tt(g["p"]),
                            g["mask_trunked"].float())).float()[0][:275]
    assert _pcc(out, g["golden_qout"].float()) > 0.999
