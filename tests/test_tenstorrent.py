import torch
import numpy as np
from boltz.model.modules.tenstorrent import (
    filter_dict,
    PairformerModule,
    DiffusionTransformerModule,
    MSAModule,
)
from boltz.model.modules.trunk import PairformerModule as PairformerModuleTorch, MSAModule as MSAModuleTorch
from boltz.model.modules.diffusion import (
    DiffusionTransformer as DiffusionTransformerTorch,
)

torch.set_grad_enabled(False)
torch.manual_seed(0)

state_dict = torch.load(
    "/home/moritz/.boltz/boltz1_conf.ckpt", map_location="cpu", mmap=True
)["state_dict"]


def median_relative_error(a, b):
    return ((a - b).abs() / b.abs()).median().item()


def test_pairformer():
    pairformer = PairformerModule(
        n_blocks=2,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
        att_head_dim=24,
        att_n_heads=16,
    )
    pairformer_torch = PairformerModuleTorch(
        token_s=384, token_z=128, num_blocks=2
    ).eval()
    pairformer_state_dict = filter_dict(state_dict, "pairformer_module")
    pairformer.load_state_dict(
        pairformer_state_dict,
        strict=False,
    )
    pairformer_torch.load_state_dict(pairformer_state_dict, strict=False)
    inputs = torch.load(
        "/home/moritz/tt-metal/tt-boltz/tests/pairformer_inputs_686.pt"
    )
    s = inputs["s"]
    z = inputs["z"]
    mask = torch.ones(1, 686)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    s_tt, z_tt = pairformer(s, z, mask, pair_mask)
    s_torch, z_torch = pairformer_torch(s, z, mask, pair_mask)
    print(median_relative_error(s_tt, s_torch), median_relative_error(z_tt, z_torch))


def test_token_transformer():
    token_transformer = DiffusionTransformerModule(
        n_layers=24,
        dim=768,
        n_heads=16,
    )
    token_transformer_torch = DiffusionTransformerTorch(
        depth=24, heads=16, dim=768, dim_single_cond=768, dim_pairwise=128
    ).eval()
    token_transformer_state_dict = filter_dict(
        state_dict, "structure_module.score_model.token_transformer"
    )
    token_transformer.load_state_dict(
        token_transformer_state_dict,
        strict=False,
    )
    token_transformer_torch.load_state_dict(token_transformer_state_dict, strict=False)
    inputs = torch.load("/home/moritz/tt-metal/tt-boltz/tests/diffusion_inputs_686.pt")
    a = inputs["a"]
    s = inputs["s"]
    z = inputs["z"]
    mask = torch.ones(1, 686)
    a_tt = token_transformer(
        a,
        s,
        z,
        mask,
    )
    a_torch = token_transformer_torch(
        a,
        s,
        z,
        mask,
    )
    print(
        median_relative_error(a_tt, a_torch),
    )

def test_msa():
    msa = MSAModule(
        n_blocks=4,
        avg_head_dim=32,
        avg_n_heads=8,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
    )
    msa_torch = MSAModuleTorch(msa_s=64, token_z=128, s_input_dim=455, msa_blocks=4, msa_dropout=0, z_dropout=0).eval()
    msa_state_dict = filter_dict(state_dict, "msa_module")
    msa.load_state_dict(msa_state_dict)
    msa_torch.load_state_dict(msa_state_dict)
    inputs = torch.load("/home/moritz/tt-metal/tt-boltz/tests/msa_inputs_prot.pt")
    z = inputs["z"]
    emb = inputs["emb"]
    feats = inputs["feats"]
    z_tt = msa(z, emb, feats)
    z_torch = msa_torch(z, emb, feats)
    print(median_relative_error(z_tt, z_torch))
