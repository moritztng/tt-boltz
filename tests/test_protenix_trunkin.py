"""On-device parity for the Protenix-v2 trunk input (s_inputs -> s_init, z_init)."""
import os, pickle, pytest, ttnn
_GOLD = os.path.expanduser("~/protenix_trunkin_gold.pkl")
pytestmark = pytest.mark.skipif(not os.path.exists(_GOLD), reason="run scripts/protenix_extract_trunkin.py")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def test_trunk_input_on_device():
    from tt_bio.tenstorrent import get_device
    from tt_bio.protenix import TrunkInput
    g = pickle.load(open(_GOLD, "rb"))
    dev = get_device()
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    def tt(x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    N = g["s_inputs"].shape[0]
    ti = TrunkInput(g["weights"], ck)
    s_init, z_init = ti(tt(g["s_inputs"]), tt(g["relp"]), tt(g["token_bonds"].unsqueeze(-1)))
    si = ttnn.to_torch(s_init).float()[:N]; zi = ttnn.to_torch(z_init).float()[:N, :N]
    assert _pcc(si, g["golden_s_init"].float()) > 0.999
    assert _pcc(zi, g["golden_z_init"].float()) > 0.999
