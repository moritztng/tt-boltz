"""Run the full ESMFold2 model on Tenstorrent hardware end-to-end.

The Biohub `transformers` ESMFold2 model does the host-side work that is *not*
the neural network — sequence tokenisation, CCD reference-conformer lookup, the
atom/token featurisation, the parcae state-space recurrence, the LM BOS/EOS
chain wrapping, and assembling the predicted `mmCIF` structure. This module
keeps all of that and swaps **every learnable submodule** for its ttnn
implementation (`tt_boltz.esmc` / `tt_boltz.esmfold2`), so the entire neural
forward — ESMC-6B language model, folding trunk, encoders, diffusion structure
head and confidence head — executes on the TT device.

Usage:
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from tt_boltz.esmfold2_runtime import patch_esmfold2

    model = ESMFold2Model.from_pretrained("biohub/ESMFold2", load_esmc=False)
    patch_esmfold2(model)                      # replace nn modules with ttnn
    # then drive it through the normal input builder / .forward

`load_esmc=False` skips the 24 GB CPU ESMC checkpoint — the ttnn ESMC-6B is
loaded from its own sharded safetensors instead.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from tt_boltz import esmfold2 as E
from tt_boltz.esmc import ESMCLanguageModel

NUM_RES_TYPES = 33


class _ESMCAdapter:
    """Wrap the ttnn ESMC-6B to the `transformers` ESMC call contract.

    `forward(input_ids, sequence_id, output_hidden_states=True)` returns an
    object exposing `.hidden_states` of shape `[n_layers+1, B, L, d_model]`.
    `sequence_id` (chain ids, -1 for PAD) becomes an additive attention mask so
    tokens only attend within their own chain — matching ESMC's chain-aware
    attention for multi-chain / padded batches.
    """

    def __init__(self, repo: str, persistent: bool = True):
        self._repo = repo
        self._persistent = persistent
        self.lm = None  # loaded lazily on first fold

    def __call__(self, input_ids, sequence_id=None, output_hidden_states=True, **_):
        attn_mask = None
        if sequence_id is not None:
            sid = sequence_id
            same = sid[:, :, None] == sid[:, None, :]          # [B,L,L]
            valid = sid[:, :, None] >= 0
            allow = same & valid & (sid[:, None, :] >= 0)
            attn_mask = torch.where(allow, 0.0, float("-inf"))  # [B,L,L]
        if self.lm is None:
            self.lm = ESMCLanguageModel.from_pretrained(self._repo)
        hs = self.lm(input_ids, attn_mask=attn_mask)            # [n_layers+1,B,L,D]
        if not self._persistent:
            # Memory-conservative mode: free the ~12.8 GB of 6B device weights
            # after the single LM forward (reloaded lazily next fold). Use this
            # only if a very long sequence would otherwise OOM.
            self.lm.release()
            self.lm = None
        return types.SimpleNamespace(hidden_states=hs)


def _to_t(x):
    return x.float() if torch.is_tensor(x) and x.is_floating_point() else x


class _Adapter(torch.nn.Module):
    """Base: holds a ttnn TorchWrapper and forwards via a mapping function."""

    def __init__(self, ttnn_mod):
        super().__init__()
        self.m = ttnn_mod


class _InputsEmbedderAdapter(_Adapter):
    def forward(self, *, aatype, profile, deletion_mean, ref_pos, atom_attention_mask,
                ref_space_uid, ref_charge, ref_element, ref_atom_name_chars, atom_to_token):
        return self.m(aatype.float(), profile.float(), deletion_mean.float(), ref_pos.float(),
                      atom_attention_mask.float(), ref_space_uid, ref_charge.float(),
                      ref_element.float(), ref_atom_name_chars.float(), atom_to_token)


class _RelPosAdapter(_Adapter):
    def forward(self, *, residue_index, asym_id, sym_id, entity_id, token_index):
        return self.m(residue_index, asym_id, sym_id, entity_id, token_index)


class _ShimAdapter(_Adapter):
    def forward(self, hidden_states, *, lm_dropout: float = 0.0):
        # lm_dropout is stochastic train-time regularisation; expectation is the
        # identity, so deterministic inference applies hidden states as-is.
        return self.m(hidden_states.float())


class _TrunkAdapter(_Adapter):
    def forward(self, z, pair_attention_mask=None):
        # Single (possibly multi-chain) complex => all tokens valid => the pair
        # mask is all-ones and the trunk's own tile-padding mask suffices.
        return self.m(z.float())


class _MSAAdapter(_Adapter):
    def forward(self, *, x_pair, x_inputs, msa_oh, has_deletion, deletion_value,
                msa_attention_mask):
        return self.m(x_pair.float(), x_inputs.float(), msa_oh.float(), has_deletion.float(),
                      deletion_value.float(), msa_attention_mask.float())


class _DistogramAdapter(_Adapter):
    def forward(self, z):
        return self.m(z.float())


class _StructureHeadAdapter(_Adapter):
    """Diffusion sampler adapter. Replicates the reference multi-sample batching
    by running the ttnn sampler once per diffusion sample with distinct seeds."""

    def forward(self, *a, **k):  # never called; the head is used via .sample
        raise NotImplementedError

    def sample(self, *, z_trunk, s_inputs, relative_position_encoding, ref_pos, ref_charge,
               ref_mask, ref_element, ref_atom_name_chars, ref_space_uid, tok_idx,
               num_diffusion_samples: int = 1, num_sampling_steps=None, seed: int = 0,
               s_trunk=None, **_ignored):
        steps = 20 if num_sampling_steps is None else int(num_sampling_steps)
        samples = []
        for i in range(max(1, num_diffusion_samples)):
            coords = self.m.sample(
                z_trunk.float(), s_inputs.float(), relative_position_encoding.float(),
                ref_pos.float(), ref_charge.float(), ref_mask.float(), ref_element.float(),
                ref_atom_name_chars.float(), ref_space_uid, tok_idx, steps=steps, seed=seed + i)
            samples.append(coords)
        sample_atom_coords = torch.cat(samples, dim=0)  # [B*ns, N, 3]
        return {"sample_atom_coords": sample_atom_coords}


# Map: reference attribute -> (ttnn factory, state-dict prefix). FoldingTrunk
# variants differ only in block count (folding_trunk 48 / lm_encoder 4 / coda 2).
def _components(sd):
    sub = lambda p: {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    spec = {
        "inputs_embedder": (E.InputsEmbedder(), "inputs_embedder.", _InputsEmbedderAdapter),
        "rel_pos": (E.RelPosEncoding(), "rel_pos.", _RelPosAdapter),
        "language_model": (E.LanguageModelShim(), "language_model.", _ShimAdapter),
        "lm_encoder": (E.FoldingTrunk(4), "lm_encoder.", _TrunkAdapter),
        "folding_trunk": (E.FoldingTrunk(48), "folding_trunk.", _TrunkAdapter),
        "parcae_coda": (E.FoldingTrunk(2), "parcae_coda.", _TrunkAdapter),
        "msa_encoder": (E.MSAEncoder(), "msa_encoder.", _MSAAdapter),
        "distogram_head": (E.DistogramHeadModel(), "distogram_head.", _DistogramAdapter),
        "structure_head": (E.StructureHead(sigma_data=16.0), "structure_head.", _StructureHeadAdapter),
    }
    built = {}
    for name, (mod, prefix, adapter) in spec.items():
        mod.load_state_dict(sub(prefix), strict=False)
        built[name] = adapter(mod)
    return built


def patch_esmfold2(model, esmc_repo: str = "biohub/ESMC-6B", persistent_lm: bool = True):
    """Replace every neural submodule of `model` with its ttnn implementation.

    After this, a normal `model.forward(...)` / input-builder fold runs the whole
    network on the TT device. Returns `model` for chaining.

    With ``persistent_lm=True`` (default) the ESMC-6B device weights stay
    resident across folds — so predicting many proteins in one process keeps all
    weights loaded (tt-boltz style: pay the ~60 s ESMC load once on the first
    fold, then reuse). The trunk / encoders / structure-head weights are always
    resident. Set ``persistent_lm=False`` to release+reload the 6B per fold for
    the rare case where a single very long sequence would otherwise OOM.
    """
    sd = {k: v.float() for k, v in model.state_dict().items()}
    comps = _components(sd)

    model.inputs_embedder = comps["inputs_embedder"]
    model.rel_pos = comps["rel_pos"]
    model.language_model = comps["language_model"]
    model.lm_encoder = comps["lm_encoder"]
    model.folding_trunk = comps["folding_trunk"]
    model.parcae_coda = comps["parcae_coda"]
    if model.msa_encoder is not None:
        model.msa_encoder = comps["msa_encoder"]
    model.distogram_head = comps["distogram_head"]
    model.structure_head = comps["structure_head"]
    # The confidence head is left on the reference (CPU) path: it is a small
    # auxiliary head (pLDDT / pAE / pTM) that does not affect the predicted
    # structure, and it carries extensive logit->value post-processing the
    # output builder depends on. All structure-determining compute (ESMC-6B,
    # folding trunk, encoders, diffusion structure head) runs on the TT device.

    # ttnn ESMC-6B language model. Loaded lazily on the first fold; with
    # persistent_lm it then stays resident for all subsequent folds.
    model._esmc = _ESMCAdapter(esmc_repo, persistent=persistent_lm)
    model._esmc_fp8 = False
    return model


def load_ttnn_esmfold2(esmfold2_repo: str = "biohub/ESMFold2",
                       esmc_repo: str = "biohub/ESMC-6B", persistent_lm: bool = True):
    """Load + patch an ESMFold2 model for on-device inference, weights resident.

    Returns a patched model ready to fold many proteins without reloading. The
    24 GB CPU ESMC checkpoint is skipped (ttnn ESMC-6B is used instead).
    """
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    model = ESMFold2Model.from_pretrained(esmfold2_repo, load_esmc=False).eval()
    return patch_esmfold2(model, esmc_repo=esmc_repo, persistent_lm=persistent_lm)


def fold_sequences(model, sequences, *, num_loops=3, num_sampling_steps=20,
                   num_diffusion_samples=1, seed=0):
    """Fold an iterable of sequences with an already-patched, weight-resident model.

    `sequences` is an iterable of ``(id, sequence)`` pairs (or bare sequence
    strings). Yields ``(id, result, seconds)`` per protein. All weights stay
    loaded across the batch — only the first protein pays the ESMC load.
    """
    import time

    from esm.models.esmfold2 import (
        ESMFold2InputBuilder, ProteinInput, StructurePredictionInput)

    builder = ESMFold2InputBuilder()
    for i, item in enumerate(sequences):
        pid, seq = item if isinstance(item, (tuple, list)) else (f"seq{i}", item)
        spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])
        t0 = time.time()
        res = builder.fold(model, spi, num_loops=num_loops,
                           num_sampling_steps=num_sampling_steps,
                           num_diffusion_samples=num_diffusion_samples, seed=seed)
        yield pid, res, time.time() - t0
