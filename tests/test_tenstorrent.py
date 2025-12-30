import pytest, torch
from boltz.model.modules.tenstorrent import (
    filter_dict,
    PairformerModule,
    MSAModule,
    DiffusionModule,
)
from boltz.model.modules.trunkv2 import MSAModule as MSAModuleTorch
from boltz.model.modules.diffusionv2 import DiffusionModule as DiffusionModuleTorch
from boltz.model.layers.pairformer import (
    PairformerModule as PairformerModuleTorch,
    PairformerNoSeqModule as PairformerNoSeqModuleTorch,
)
from boltz.model.modules.encoders import get_indexing_matrix, single_to_keys
from functools import partial

torch.set_grad_enabled(False)
torch.manual_seed(893)

state_dict = torch.load(
    "/home/moritz/.boltz/boltz2_conf.ckpt",
    map_location="cpu",
    mmap=True,
    weights_only=False,
)["state_dict"]


def median_relative_error(a, b):
    return ((a - b).abs() / b.abs()).median().item()


@pytest.mark.parametrize("seq_len", [100, 500, 1000])
def test_pairformer(seq_len):
    pairformer = PairformerModule(
        n_blocks=2,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
        att_head_dim=24,
        att_n_heads=16,
        transform_s=True,
    )
    pairformer_torch = PairformerModuleTorch(
        token_s=384, token_z=128, num_blocks=2, v2=True
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
def test_affinity_pairformer(seq_len):
    state_dict = torch.load(
        "/home/moritz/.boltz/boltz2_aff.ckpt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )["state_dict"]
    pairformer = PairformerModule(
        n_blocks=4,
        tri_att_head_dim=32,
        tri_att_n_heads=4,
        att_head_dim=None,
        att_n_heads=None,
        transform_s=False,
    )
    pairformer_torch = PairformerNoSeqModuleTorch(
        token_z=128, num_blocks=4, v2=True
    ).eval()
    pairformer_state_dict = filter_dict(state_dict, "affinity_module1.pairformer_stack")
    pairformer.load_state_dict(
        pairformer_state_dict,
        strict=False,
    )
    pairformer_torch.load_state_dict(pairformer_state_dict, strict=False)
    z = 26 * torch.randn(1, seq_len, seq_len, 128)
    mask = torch.ones(1, seq_len)
    mask[0, : seq_len // 2] = 0
    mask = mask[:, torch.randperm(seq_len)]
    mask = mask[:, :, None] * mask[:, None, :]
    z_tt = pairformer(None, z, mask)[1]
    z_torch = pairformer_torch(z, pair_mask=mask)
    assert median_relative_error(z_tt, z_torch) < 1e-1, "z not accurate"


def test_diffusion():
    diffusion = DiffusionModule()
    diffusion_torch = DiffusionModuleTorch(384, 128, token_transformer_heads=16).eval()
    diffusion_state_dict = filter_dict(state_dict, "structure_module.score_model")
    diffusion.load_state_dict(
        diffusion_state_dict,
        strict=False,
    )
    diffusion_torch.load_state_dict(
        diffusion_state_dict,
        strict=False,
    )
    diffusion_samples=1
    r_noisy = torch.randn(diffusion_samples, 928, 3)
    times = torch.randn(diffusion_samples)
    s_inputs = torch.randn(1, 117, 384)
    s_trunk = torch.randn(1, 117, 384)
    q = torch.randn(1, 928, 128)
    c = torch.randn(1, 928, 128)
    bias_encoder = torch.randn([1, 29, 32, 128, 12])
    bias_decoder = torch.randn([1, 29, 32, 128, 12])
    bias_token = torch.randn([1, 117, 117, 384])
    mask_atom = torch.ones([1, 928])
    mask_token = torch.ones([1, 117])
    atom_to_token = torch.ones([1, 928, 117])
    keys_indexing = get_indexing_matrix(29, 32, 128, "cpu")
    r_update = diffusion(
        r_noisy,
        times,
        s_inputs,
        s_trunk,
        q,
        c,
        bias_encoder,
        bias_token,
        bias_decoder,
        keys_indexing,
        mask_atom,
        atom_to_token,
    )
    r_update_torch = diffusion_torch(
        r_noisy=r_noisy,
        times=times,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        diffusion_conditioning={
            "q": q,
            "c": c,
            "atom_enc_bias": bias_encoder,
            "token_trans_bias": bias_token,
            "atom_dec_bias": bias_decoder,
            "to_keys": partial(
                single_to_keys, indexing_matrix=keys_indexing, W=32, H=128
            ),
        },
        feats={
            "atom_pad_mask": mask_atom,
            "atom_to_token": atom_to_token,
            "ref_pos": torch.randn(1, 928, 3),
            "token_pad_mask": mask_token,
        },
        multiplicity=diffusion_samples,
    )
    assert (
        median_relative_error(r_update, r_update_torch) < 1e-1
    ), "r_update not accurate"


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
    msa_torch = MSAModuleTorch(
        msa_s=64, token_z=128, token_s=384, msa_blocks=4, msa_dropout=0, z_dropout=0
    ).eval()
    msa_state_dict = filter_dict(state_dict, "msa_module")
    msa.load_state_dict(msa_state_dict)
    msa_torch.load_state_dict(msa_state_dict)
    z = 7 * torch.randn(1, seq_len, seq_len, 128)
    emb = torch.ones(1, seq_len, 384)
    feats = {
        "msa": torch.randint(33, (1, n_sequences, seq_len)),
        "has_deletion": torch.zeros((1, n_sequences, seq_len), dtype=torch.bool),
        "deletion_value": torch.zeros((1, n_sequences, seq_len)),
        "msa_paired": torch.zeros((1, n_sequences, seq_len)),
        "msa_mask": torch.ones((1, n_sequences, seq_len)),
        "token_pad_mask": torch.ones((1, seq_len)),
    }
    z_tt = msa(z, emb, feats)
    z_torch = msa_torch(z, emb, feats)
    assert median_relative_error(z_tt, z_torch) < 1e-1, "z not accurate"
