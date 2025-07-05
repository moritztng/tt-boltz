import pytest, torch
from boltz.model.modules.tenstorrent import (
    filter_dict,
    PairformerModule,
    DiffusionTransformerModule,
    MSAModule,
)
from boltz.model.modules.trunkv2 import MSAModule as MSAModuleTorch
from boltz.model.modules.transformersv2 import (
    DiffusionTransformer as DiffusionTransformerTorch
)
from boltz.model.layers.pairformer import PairformerModule as PairformerModuleTorch

torch.set_grad_enabled(False)
torch.manual_seed(893)

state_dict = torch.load(
    "/home/moritz/.boltz/boltz2_conf.ckpt", map_location="cpu", mmap=True
)["state_dict"]


def median_relative_error(a, b):
    return ((a - b).abs() / b.abs()).median().item()


@pytest.mark.parametrize("seq_len", [100, 500, 768])
def test_pairformer(seq_len):
    pairformer = PairformerModule(
        n_blocks=5,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
        att_head_dim=24,
        att_n_heads=16,
        transform_s=True,
    )
    pairformer_torch = PairformerModuleTorch(
        token_s=384, token_z=128, num_blocks=5, v2=True
    ).eval()
    pairformer_state_dict = filter_dict(state_dict, "pairformer_module")
    pairformer.load_state_dict(
        pairformer_state_dict,
        strict=False,
    )
    pairformer_torch.load_state_dict(pairformer_state_dict, strict=False)
    s = 8 * torch.randn(1, seq_len, 384)
    z = 26 * torch.randn(1, seq_len, seq_len, 128)
    mask = torch.ones(1, seq_len)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    s_tt, z_tt = pairformer(s, z, mask, pair_mask)
    s_torch, z_torch = pairformer_torch(s, z, mask, pair_mask)
    assert median_relative_error(s_tt, s_torch) < 1e-1, "s not accurate"
    assert median_relative_error(z_tt, z_torch) < 1e-1, "z not accurate"


@pytest.mark.parametrize("seq_len", [100, 500, 1000])
def test_token_transformer(seq_len):
    n_layers = 24
    token_transformer = DiffusionTransformerModule( 
        n_layers=n_layers,
        dim=768,
        n_heads=16,
    )
    token_transformer_torch = DiffusionTransformerTorch(
        depth=n_layers, heads=16, dim=768, dim_single_cond=768
    ).eval()
    token_transformer_state_dict = filter_dict(
        state_dict, "structure_module.score_model.token_transformer"
    )
    token_transformer.load_state_dict(
        token_transformer_state_dict,
        strict=False,
    )
    token_transformer_torch.load_state_dict(token_transformer_state_dict, strict=False)
    a = 3 + 5 * torch.randn(1, seq_len, 768)
    s = -2 + 42 * torch.randn(1, seq_len, 768)
    z = 10 * torch.randn(1, seq_len, seq_len, n_layers * 16)
    mask = torch.ones(1, seq_len)
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
    assert median_relative_error(a_tt, a_torch) < 1e-1, "a not accurate"

@pytest.mark.parametrize("seq_len", [100, 500, 1000])
def test_msa(seq_len):
    n_sequences = 100
    msa = MSAModule(
        n_blocks=4,
        avg_head_dim=32,
        avg_n_heads=8,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
    )
    msa_torch = MSAModuleTorch(msa_s=64, token_z=128, token_s=384, msa_blocks=4, msa_dropout=0, z_dropout=0).eval()
    msa_state_dict = filter_dict(state_dict, "msa_module")
    msa.load_state_dict(msa_state_dict)
    msa_torch.load_state_dict(msa_state_dict)
    z = 7 * torch.randn(1, seq_len, seq_len, 128)
    emb = torch.ones(1, seq_len, 384)
    feats = {"msa": torch.randint(33, (1, n_sequences, seq_len)),
             "has_deletion": torch.zeros((1, n_sequences, seq_len), dtype=torch.bool),
             "deletion_value": torch.zeros((1, n_sequences, seq_len)),
             "msa_paired": torch.zeros((1, n_sequences, seq_len)),
             "msa_mask": torch.ones((1, n_sequences, seq_len)),
             "token_pad_mask": torch.ones((1, seq_len)),
    }
    z_tt = msa(z, emb, feats)
    z_torch = msa_torch(z, emb, feats)
    assert median_relative_error(z_tt, z_torch) < 1e-1, "z not accurate"
