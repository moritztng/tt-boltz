import pytest, torch
from torch.nn.functional import mse_loss
from boltz.model.modules.tenstorrent import filter_dict, PairformerModule
from boltz.model.modules.trunk import PairformerModule as PairformerModuleTorch
torch.set_grad_enabled(False)
torch.manual_seed(0)
#ttnn.enable_program_cache(device)

state_dict = torch.load(
    "/home/moritz/.boltz/boltz1.ckpt", map_location="cpu", mmap=True
)["state_dict"]

def test_pairformer():
    pairformer = PairformerModule(1, 32, 4, 24, 16)
    pairformer_torch = PairformerModuleTorch(384, 128, 1).eval()
    pairformer_state_dict = filter_dict(state_dict, "pairformer_module")
    pairformer.load_state_dict(
        pairformer_state_dict,
        strict=False,
    )
    pairformer_torch.load_state_dict(pairformer_state_dict, strict=False)
    s = torch.randn(1, 128, 384)
    z = torch.randn(1, 128, 128, 128)
    mask = torch.ones(1, 128)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    s_tt, z_tt = pairformer(s, z, mask, pair_mask)
    s_torch, z_torch = pairformer_torch(s, z, mask, pair_mask)
    print(mse_loss(s_tt, s_torch), mse_loss(z_tt, z_torch))
