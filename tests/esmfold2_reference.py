"""Faithful PyTorch reference for ESMFold2 components, from the Biohub fork.

ESMFold2's architecture is NOT in the esm repo or upstream transformers — it
lives in a Biohub fork of transformers (esm pyproject pins
``transformers @ git+https://github.com/Biohub/transformers.git@main``), cloned
to ``/home/ttuser/biohub-transformers``. The neural-net building blocks are in
``src/transformers/models/esmfold2/modeling_esmfold2_common.py``.

Importing through the ``transformers`` package triggers its heavy framework
init (regex, tokenizers, ...). The folding building blocks, however, only need
torch + numpy. So we load that one file *directly* under fake namespace
packages, stubbing the two relative deps the blocks don't actually use
(``configuration_esmfold2``: only the top-level model needs the Config object;
``kernels``: optional Triton/CUDA kernels). No shared-env mutation.

This mirrors tests/esmc_reference.py and is the golden reference our ttnn
ESMFold2 port is parity-tested against.
"""

from __future__ import annotations

import importlib.util
import sys
import types

FORK_ESMFOLD2 = "/home/ttuser/biohub-transformers/src/transformers/models/esmfold2"
_MODNAME = "transformers.models.esmfold2.modeling_esmfold2_common"


def _load_common():
    if _MODNAME in sys.modules:
        return sys.modules[_MODNAME]

    # Fake parent packages so relative imports resolve without running the real
    # transformers __init__.
    for name in ("transformers", "transformers.models", "transformers.models.esmfold2"):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = []  # mark as package
            sys.modules[name] = pkg

    # Stub the relative deps the folding blocks don't need.
    cfg = types.ModuleType("transformers.models.esmfold2.configuration_esmfold2")
    cfg.ESMFold2Config = type("ESMFold2Config", (), {})
    sys.modules[cfg.__name__] = cfg
    # Empty kernels module -> `from .kernels import X` raises ImportError ->
    # the module's try/except sets TRITON_KERNELS_AVAILABLE = False.
    sys.modules["transformers.models.esmfold2.kernels"] = types.ModuleType(
        "transformers.models.esmfold2.kernels"
    )

    spec = importlib.util.spec_from_file_location(
        _MODNAME, f"{FORK_ESMFOLD2}/modeling_esmfold2_common.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODNAME] = mod
    spec.loader.exec_module(mod)
    return mod


common = _load_common()

# Folding-trunk config (from biohub/ESMFold2 weights: 48 blocks, c_z=256,
# pair_transition FFN hidden 1024 => expansion_ratio 4).
FOLDING_TRUNK = dict(n_layers=48, d_pair=256, expansion_ratio=4)


# Token DiffusionTransformer config (DiffusionModule: c_token=768, c_z=256,
# token_num_heads=16, token_num_blocks=12, d_cond=c_token, transition_multiplier=2).
DIFFUSION_TOKEN = dict(
    d_model=768, d_pair=256, num_heads=16, num_blocks=12, d_cond=768,
    transition_multiplier=2, use_conditioning=True,
)


def make_diffusion_transformer(num_blocks: int | None = None, seed: int = 0):
    """Reference token DiffusionTransformer (random init)."""
    import torch

    torch.manual_seed(seed)
    cfg = dict(DIFFUSION_TOKEN)
    if num_blocks is not None:
        cfg["num_blocks"] = num_blocks
    return common.DiffusionTransformer(**cfg).eval()


DIFFUSION_COND = dict(c_z=256, c_s=768, c_s_inputs=451, sigma_data=16.0, fourier_dim=256, transition_multiplier=2)


def make_diffusion_conditioning(seed: int = 0):
    """Reference DiffusionConditioning (random init)."""
    import torch

    torch.manual_seed(seed)
    return common.DiffusionConditioning(**DIFFUSION_COND).eval()


SWA_ATOM = dict(
    d_atom=128, n_blocks=3, n_heads=4, swa_window_size=128, expansion_ratio=2,
    spatial_rope_base_frequency=20.0, n_spatial_rope_pairs_per_axis=2,
    n_uid_rope_pairs=10, uid_rope_base_frequency=10000.0,
)


def make_swa_atom_transformer(n_blocks: int | None = None, seed: int = 0):
    """Reference SWAAtomTransformer (random init)."""
    import torch

    torch.manual_seed(seed)
    cfg = dict(SWA_ATOM)
    if n_blocks is not None:
        cfg["n_blocks"] = n_blocks
    return common.SWAAtomTransformer(**cfg).eval()


DIFFUSION_MODULE = dict(
    c_atom=128, c_token=768, c_z=256, c_s_inputs=451, sigma_data=16.0, fourier_dim=256,
    atom_num_blocks=3, atom_num_heads=4, token_num_blocks=12, token_num_heads=16,
    transition_multiplier=2, swa_window_size=128, spatial_rope_base_frequency=20.0,
    n_spatial_rope_pairs_per_axis=2, n_uid_rope_pairs=10, uid_rope_base_frequency=10000.0,
)

# Params that are zero/constant-initialized in the reference; randomize them so
# parity tests exercise the gated/modulated paths (loaded identically into ttnn).
_ZERO_INIT_KEYS = ("out_gate", "output_gate", "s_to_token", "adaln_modulation")


def make_diffusion_module(seed: int = 0):
    """Reference DiffusionModule (random init; zero-init gates perturbed)."""
    import torch

    torch.manual_seed(seed)
    m = common.DiffusionModule(**DIFFUSION_MODULE).eval()
    for name, p in m.named_parameters():
        if any(k in name for k in _ZERO_INIT_KEYS):
            torch.nn.init.normal_(p, std=0.1)
    return m


def make_lm_shim(seed: int = 0, d_model: int = 2560, num_layers: int = 80):
    """Reference LanguageModelShim."""
    import torch

    torch.manual_seed(seed)
    return common.LanguageModelShim(d_z=256, d_model=d_model, num_layers=num_layers).eval()


def make_msa_encoder(seed: int = 0):
    """Self-contained reference MSAEncoder (matches modeling_esmfold2.MSAEncoder)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class PairTransitionRef(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.ffn = common.SwiGLUMLP(d, expansion_ratio=4, bias=False)

        def forward(self, x):
            return self.ffn(self.norm(x))

    class BlockRef(nn.Module):
        def __init__(self, is_final):
            super().__init__()
            self.is_final = is_final
            self.outer_product_mean = common.OuterProductMean(128, 32, 256)
            if not is_final:
                self.msa_pair_weighted_averaging = common.MSAPairWeightedAveraging(128, 256, 8, 16)
                self.msa_transition = PairTransitionRef(128)
            self.tri_mul_out = common.TriangleMultiplicativeUpdate(dim=256, _outgoing=True)
            self.tri_mul_in = common.TriangleMultiplicativeUpdate(dim=256, _outgoing=False)
            self.tri_mul_out.set_chunk_size(None); self.tri_mul_in.set_chunk_size(None)
            self.pair_transition = PairTransitionRef(256)

        def forward(self, m, pair, msa_mask, pair_mask):
            pair = pair + self.outer_product_mean(m, msa_mask)
            if not self.is_final:
                m = m + self.msa_pair_weighted_averaging(m, pair, pair_mask)
                m = m + self.msa_transition(m)
            pair = pair + self.tri_mul_out(pair, mask=pair_mask)
            pair = pair + self.tri_mul_in(pair, mask=pair_mask)
            pair = pair + self.pair_transition(pair)
            return m, pair

    class MSAEncoderRef(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(35, 128, bias=False)
            self.project_inputs = nn.Linear(451, 128, bias=False)
            self.blocks = nn.ModuleList([BlockRef(i == 3) for i in range(4)])

        def forward(self, x_pair, x_inputs, msa_oh, has_deletion, deletion_value, msa_mask):
            m_feat = torch.cat([msa_oh, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1)
            m = self.embed(m_feat) + self.project_inputs(x_inputs).unsqueeze(2)
            tok_mask = msa_mask[:, :, 0].bool()
            pair_mask = tok_mask.unsqueeze(2) & tok_mask.unsqueeze(1)
            for block in self.blocks:
                m, x_pair = block(m, x_pair, msa_mask, pair_mask)
            return x_pair

    return MSAEncoderRef().eval()


def make_relpos(seed: int = 0):
    """Reference relative-position encoding."""
    import torch

    torch.manual_seed(seed)
    return common.ResIdxAsymIdSymIdEntityIdEncoding(
        n_relative_residx_bins=32, n_relative_chain_bins=2, d_pair=256
    ).eval()


def make_inputs_embedder(seed: int = 0):
    """Reference inputs embedder (atom encoder, structure_prediction=False) + concat."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class InputsEmbedderRef(nn.Module):
        def __init__(self):
            super().__init__()
            self.atom_attention_encoder = common.ESMFold2AtomEncoder(
                d_atom=128, d_token=768, n_blocks=3, n_heads=4, swa_window_size=128,
                expansion_ratio=2, structure_prediction=False,
                spatial_rope_base_frequency=20.0, n_spatial_rope_pairs_per_axis=2,
                n_uid_rope_pairs=10, uid_rope_base_frequency=10000.0,
            )

        def forward(self, aatype, profile, deletion_mean, ref_pos, atom_mask,
                    ref_space_uid, ref_charge, ref_element, ref_atom_name_chars, atom_to_token):
            a, _q, _c, _p, _i = self.atom_attention_encoder(
                ref_pos=ref_pos, atom_attention_mask=atom_mask, ref_space_uid=ref_space_uid,
                ref_charge=ref_charge, ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars, atom_to_token=atom_to_token,
            )
            return torch.cat([a, aatype, profile, deletion_mean.unsqueeze(-1)], dim=-1)

    return InputsEmbedderRef().eval()


def make_distogram_head(seed: int = 0):
    """Reference model-level distogram head: Linear(d_pair, 64)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    return nn.Linear(256, 64).eval()


def make_confidence_core(seed: int = 0):
    """Self-contained reference for the ConfidenceHead learned core (4 logits).

    Composes the same submodules the real ConfidenceHead uses and replicates the
    logit-producing forward (pae/pde/plddt/resolved). Matches the ttnn port; real
    checkpoint weights load by the same key names.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)

    class ConfidenceCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.s_inputs_norm = nn.LayerNorm(451)
            self.z_norm = nn.LayerNorm(256)
            self.s_to_z = nn.Linear(451, 256, bias=False)
            self.s_to_z_transpose = nn.Linear(451, 256, bias=False)
            self.s_to_z_prod_in1 = nn.Linear(451, 256, bias=False)
            self.s_to_z_prod_in2 = nn.Linear(451, 256, bias=False)
            self.s_to_z_prod_out = nn.Linear(256, 256, bias=False)
            self.dist_bin_pairwise_embed = nn.Embedding(39, 256)
            self.folding_trunk = common.FoldingTrunk(n_layers=4, d_pair=256, expansion_ratio=4)
            self.folding_trunk.set_chunk_size(None)
            self.row_attention_pooling = common.RowAttentionPooling(d_pair=256, d_single=768)
            self.pae_ln = nn.LayerNorm(256); self.pae_head = nn.Linear(256, 64, bias=False)
            self.pde_ln = nn.LayerNorm(256); self.pde_head = nn.Linear(256, 64, bias=False)
            self.plddt_ln = nn.LayerNorm(768)
            self.plddt_weight = nn.Parameter(torch.randn(23, 768, 50) * 0.1)
            self.resolved_ln = nn.LayerNorm(768)
            self.resolved_weight = nn.Parameter(torch.randn(23, 768, 2) * 0.1)
            self.register_buffer("boundaries", torch.linspace(3.25, 50.75, 38))

        def forward(self, s_inputs, z, rep_coords, atom_to_token, intra_idx):
            B, L = z.shape[0], z.shape[1]
            s_n = self.s_inputs_norm(s_inputs)
            zb = self.z_norm(z)
            zb = zb + self.s_to_z(s_n).unsqueeze(2) + self.s_to_z_transpose(s_n).unsqueeze(1)
            zb = zb + self.s_to_z_prod_out(
                self.s_to_z_prod_in1(s_n)[:, :, None, :] * self.s_to_z_prod_in2(s_n)[:, None, :, :]
            )
            d = torch.cdist(rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist")
            bins = (d.unsqueeze(-1) > self.boundaries).sum(-1).long()
            pair = zb + self.dist_bin_pairwise_embed(bins)
            pair = pair + self.folding_trunk(pair)
            single = self.row_attention_pooling(pair, torch.ones(B, L))
            pae = self.pae_head(self.pae_ln(pair))
            pde = self.pde_head(self.pde_ln(pair))
            s_at = common.gather_token_to_atom(single, atom_to_token)
            plddt = torch.einsum("bac,bacd->bad", self.plddt_ln(s_at), self.plddt_weight[intra_idx])
            resolved = torch.einsum("bac,bacd->bad", self.resolved_ln(s_at), self.resolved_weight[intra_idx])
            return pae, pde, plddt, resolved

    return ConfidenceCore().eval()


def make_folding_trunk(n_layers: int | None = None, seed: int = 0):
    """Reference FoldingTrunk (random init). chunk_size=None for bit-exact parity."""
    import torch

    torch.manual_seed(seed)
    cfg = dict(FOLDING_TRUNK)
    if n_layers is not None:
        cfg["n_layers"] = n_layers
    trunk = common.FoldingTrunk(**cfg).eval()
    trunk.set_chunk_size(None)
    return trunk
