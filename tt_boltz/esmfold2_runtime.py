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

import os
import sys
import types

import torch
import torch.nn.functional as F

from tt_boltz import esmfold2 as E
from tt_boltz.esmc import ESMCLanguageModel


def _ensure_reference_on_path():
    """Put the Biohub `esm` + `transformers` (ESMFold2 fork) on sys.path.

    They supply the host-side featurization and mmCIF assembly (not the neural
    compute, which is ttnn). Locations default to the standard sibling clones
    and can be overridden with the ESM_PATH / BIOHUB_TRANSFORMERS_PATH env vars.
    """
    for env, default in [("BIOHUB_TRANSFORMERS_PATH", "/home/ttuser/biohub-transformers/src"),
                         ("ESM_PATH", "/home/ttuser/esm")]:
        path = os.environ.get(env, default)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


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

    def preload(self):
        """Load the ESMC-6B weights now (≈60 s) instead of lazily on the first
        fold — lets the CLI attribute that time to the 'loading' stage."""
        if self.lm is None:
            self.lm = ESMCLanguageModel.from_pretrained(self._repo)

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
            # after the single LM forward (reloaded lazily next fold). Not needed
            # in practice — persistent mode fits the full L<=1024 range — but
            # available as extra headroom for unusually large inputs.
            self.lm.release()
            self.lm = None
        return types.SimpleNamespace(hidden_states=hs)


def _to_t(x):
    return x.float() if torch.is_tensor(x) and x.is_floating_point() else x


class _Adapter(torch.nn.Module):
    """Bridge a reference submodule call to a ttnn TorchWrapper's positional forward.

    `argnames` lists the reference kwarg names in the order the wrapper expects;
    floating tensors are cast to fp32. With no `argnames`, the reference's first
    positional arg is forwarded (single-input modules like the trunk / shim /
    distogram head, which ignore any extra kwargs such as `pair_attention_mask`).
    """

    def __init__(self, mod, *argnames):
        super().__init__()
        self.m = mod
        self.argnames = argnames

    def forward(self, *args, **kw):
        if self.argnames:
            return self.m(*[_to_t(kw[n]) for n in self.argnames])
        return self.m(_to_t(args[0]))

    # No-ops so a reference module that owns this wrapper (e.g. the confidence
    # head owning a folding trunk) can still call these on it.
    def set_kernel_backend(self, backend):
        pass

    def set_chunk_size(self, chunk_size):
        pass


class _StructureHeadAdapter(_Adapter):
    """Diffusion sampler adapter — runs best-of-N as ONE batched B=N trajectory.

    The diffusion conditioning is molecule-only (shared across samples), and a
    single B=1 sample underutilizes the device, so the N samples run as one
    B=N pass (distinct noise per row) for ~1x the cost of one — instead of N
    serial calls. Output is [N, n_atoms, 3], the contract the confidence head
    and best-of-N selection already expect."""

    def sample(self, *, z_trunk, s_inputs, relative_position_encoding, ref_pos, ref_charge,
               ref_mask, ref_element, ref_atom_name_chars, ref_space_uid, tok_idx,
               num_diffusion_samples: int = 1, num_sampling_steps=None, seed: int = 0,
               **_ignored):
        steps = 20 if num_sampling_steps is None else int(num_sampling_steps)
        coords = self.m.sample(
            z_trunk.float(), s_inputs.float(), relative_position_encoding.float(),
            ref_pos.float(), ref_charge.float(), ref_mask.float(), ref_element.float(),
            ref_atom_name_chars.float(), ref_space_uid, tok_idx,
            steps=steps, seed=seed, multiplicity=max(1, num_diffusion_samples))
        return {"sample_atom_coords": coords}  # [N, n_atoms, 3]


# reference attribute -> (ttnn wrapper, state-dict prefix, reference kwarg order).
# An empty kwarg tuple means "forward the first positional arg" (trunk/shim/
# distogram). Block counts and MSA head width follow the checkpoint config, so
# the same code loads both ESMFold2 (48 trunk blocks, msa_head_width 16) and
# ESMFold2-Fast (24 trunk blocks, msa_head_width 32) — see _spec().
def _spec(config):
    """Build the component spec for this checkpoint's config (variant-aware)."""
    ft, lm, pc, msa = (config.folding_trunk, config.lm_encoder,
                       config.parcae, config.msa_encoder)
    spec = {
        "inputs_embedder": (lambda: E.InputsEmbedder(), "inputs_embedder.",
            ("aatype", "profile", "deletion_mean", "ref_pos", "atom_attention_mask",
             "ref_space_uid", "ref_charge", "ref_element", "ref_atom_name_chars", "atom_to_token")),
        "rel_pos": (lambda: E.RelPosEncoding(), "rel_pos.",
            ("residue_index", "asym_id", "sym_id", "entity_id", "token_index")),
        "language_model": (lambda: E.LanguageModelShim(), "language_model.", ()),
        "lm_encoder": (lambda: E.FoldingTrunk(lm.n_layers), "lm_encoder.", ()),
        "folding_trunk": (lambda: E.FoldingTrunk(ft.n_layers), "folding_trunk.", ()),
        "parcae_coda": (lambda: E.FoldingTrunk(pc.coda_n_layers), "parcae_coda.", ()),
        "distogram_head": (lambda: E.DistogramHeadModel(), "distogram_head.", ()),
        "structure_head": (lambda: E.StructureHead(sigma_data=16.0), "structure_head.", None),
    }
    # The MSA encoder exists only when enabled (ESMFold2-Fast ships without it),
    # mirroring the reference (model.msa_encoder is None when disabled). Build it
    # only then — patch_esmfold2 likewise uses it only if model.msa_encoder is set.
    if getattr(msa, "enabled", True):
        spec["msa_encoder"] = (
            lambda: E.MSAEncoder(msa.n_layers, msa.n_heads_msa, msa.msa_head_width),
            "msa_encoder.",
            ("x_pair", "x_inputs", "msa_oh", "has_deletion", "deletion_value", "msa_attention_mask"))
    return spec


def _components(sd, config):
    sub = lambda p: {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    built = {}
    for name, (factory, prefix, argnames) in _spec(config).items():
        mod = factory()
        mod.load_state_dict(sub(prefix), strict=False)
        cls = _StructureHeadAdapter if argnames is None else _Adapter
        built[name] = cls(mod, *(argnames or ()))
    return built


def _install_resident_trunk_loop(model):
    """Replace the reference `_run_one_loop` with an on-device, resident-z version.

    Two wins over the per-module reference loop:
      * Deterministic inference (the per-loop lm_dropout's expectation is the
        identity) makes the LM-encoder, MSA-encoder and injection projection
        LOOP-INVARIANT — they are computed once instead of every iteration.
      * The pair state z stays resident on the TT device across all trunk
        iterations: the parcae recurrence (a*z + inject) and the folding trunk
        both run on-device, so the ~L²·256 pair tensor is never round-tripped
        (host<->device, with tile-layout conversion) per loop.
    """
    import types

    import ttnn

    from tt_boltz import esmfold2 as E

    ftw = model.folding_trunk.m  # _Adapter.m -> E.FoldingTrunk TorchWrapper
    overwrite = bool(getattr(model.config, "msa_encoder_overwrite", True))

    def _run_one_loop(self, z, z_init, lm_z, _msa_kwargs, pair_mask, a, b_mat, total_steps):
        # --- loop-invariant injection, computed ONCE ---
        z_inject = z_init
        if self.msa_encoder is not None and _msa_kwargs is not None:
            # reference passes x_pair (the current pair state) separately from _msa_kwargs
            msa_pair = self.msa_encoder(x_pair=z_inject, **_msa_kwargs).to(z_init.dtype)
            z_inject = msa_pair if overwrite else (z_inject + msa_pair)
        if lm_z is not None and self.lm_encoder is not None:
            refined = self.lm_encoder(lm_z.to(z_init.dtype), pair_attention_mask=pair_mask)
            z_inject = z_inject + refined.to(z_init.dtype)
        injected = self.parcae_input_norm(z_inject)
        inject_proj = F.linear(injected.to(z.dtype), b_mat)  # [1,L,L,256] (host)

        # --- resident-z recurrence on device ---
        Lp = z.shape[1]
        pad = (-Lp) % E.PAD_MULTIPLE
        padz = lambda t: F.pad(t, (0, 0, 0, pad, 0, pad)) if pad else t
        mask = None
        if pad:
            real = torch.zeros(1, Lp + pad, Lp + pad)
            real[:, :Lp, :Lp] = 1.0
            mask = ftw._from_torch(real)
        zt = ftw._from_torch(padz(z).float())
        ipt = ftw._from_torch(padz(inject_proj).float())
        at = ftw._from_torch(a.reshape(1, 1, 1, -1).float())  # parcae a, broadcasts over L,L
        for _step in range(total_steps):
            E.report_progress("trunk", _step, total_steps)
            az = ttnn.multiply(zt, at)
            ttnn.deallocate(zt)
            znew = ttnn.add(az, ipt)
            ttnn.deallocate(az)
            zt = ftw.module(znew, mask)  # folding trunk consumes znew, returns new z
        z_out = ftw._to_torch(zt)[:, :Lp, :Lp, :].to(z.dtype)
        for t in (zt, ipt, at, mask):
            if t is not None:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
        return z_out

    model._run_one_loop = types.MethodType(_run_one_loop, model)


def patch_esmfold2(model, esmc_repo: str = "biohub/ESMC-6B", persistent_lm: bool = True):
    """Replace every neural submodule of `model` with its ttnn implementation.

    After this, a normal `model.forward(...)` / input-builder fold runs the whole
    network on the TT device. Returns `model` for chaining.

    With ``persistent_lm=True`` (default) the ESMC-6B device weights stay
    resident across folds — so predicting many proteins in one process keeps all
    weights loaded (tt-boltz style: pay the ~60 s ESMC load once on the first
    fold, then reuse). The trunk / encoders / structure-head weights are always
    resident. This fits the full supported range: a single Blackhole holds the
    resident 6B (~12.8 GB) plus the L=1024 trunk/diffusion activations with room
    to spare (validated: L=1024 folds in persistent mode without OOM).
    ``persistent_lm=False`` releases+reloads the 6B per fold as extra headroom
    for unusually large inputs, at the cost of an ESMC reload each fold.
    """
    sd = {k: v.float() for k, v in model.state_dict().items()}
    comps = _components(sd, model.config)

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
    # Confidence head: its dominant cost is an internal 4-block pair trunk
    # (triangle-multiplicative updates, O(L^3)) — move that onto the device too.
    # `FoldingTrunk` returns the fully residual-updated pair, matching the
    # reference, so the head's `pair.add_(folding_trunk(pair))` is unchanged. The
    # reference head keeps the cheap O(L^2) glue (s->z products, pae/pde heads,
    # row pooling) and the logit->value post-processing (pLDDT / pAE / pTM) the
    # output builder needs. pLDDT/pAE/pTM do not affect the predicted structure.
    sub = lambda p: {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    n_conf_blocks = len(model.confidence_head.folding_trunk.blocks)
    conf_trunk = E.FoldingTrunk(n_conf_blocks)
    conf_trunk.load_state_dict(sub("confidence_head.folding_trunk."), strict=False)
    model.confidence_head.folding_trunk = _Adapter(conf_trunk)

    # ttnn ESMC-6B language model. Loaded lazily on the first fold; with
    # persistent_lm it then stays resident for all subsequent folds.
    model._esmc = _ESMCAdapter(esmc_repo, persistent=persistent_lm)
    model._esmc_fp8 = False

    # Keep the pair state resident on-device across the trunk loop (hoist the
    # loop-invariant LM/MSA/injection work and run the parcae recurrence on
    # device), avoiding per-loop host<->device round-trips of the L²·256 pair.
    _install_resident_trunk_loop(model)
    return model


def load_ttnn_esmfold2(esmfold2_repo: str = "biohub/ESMFold2",
                       esmc_repo: str = "biohub/ESMC-6B", persistent_lm: bool = True,
                       fast: bool = False):
    """Load + patch an ESMFold2 model for on-device inference, weights resident.

    Returns a patched model ready to fold many proteins without reloading. The
    24 GB CPU ESMC checkpoint is skipped (ttnn ESMC-6B is used instead).

    `fast` (the CLI `--fast` flag, off by default — same opt-in semantics as the
    Boltz-2 path) runs the heavy matmuls in block-fp8 (bfloat8_b) for a faster
    fold at a slight precision cost:
      * folding-trunk triangle-multiplications (the dominant O(L^3) cost),
      * the ESMC-6B projection/FFN weights (qkv, out_proj, fc1, fc2) — which also
        halves the resident language-model size (~12.8 GB -> ~6.4 GB),
      * the pair-transition FFN (shared SwiGLU).
    The token-DiT attention stays fp32 and the diffusion coords stay bf16
    regardless, so the structure head's precision is unaffected. Default (off)
    is full bf16/fp32 precision.
    """
    from tt_boltz import tenstorrent
    tenstorrent.set_fast_mode(fast)
    _ensure_reference_on_path()
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    model = ESMFold2Model.from_pretrained(esmfold2_repo, load_esmc=False).eval()
    return patch_esmfold2(model, esmc_repo=esmc_repo, persistent_lm=persistent_lm)


def resolve_msa(msa_spec, sequence, msa_dir=None, max_sequences=16384):
    """Resolve a chain's MSA to an esm ``MSA`` object (or None).

    Tries, in order: an explicit a3m path (``msa_spec``); a cached
    ``{sha256(seq)[:16]}.a3m`` in ``msa_dir`` (written by the predict driver
    after a server / local-DB search). Returns None for single-sequence folding.
    """
    _ensure_reference_on_path()
    import hashlib
    from pathlib import Path

    from esm.utils.msa.msa import MSA

    candidates = []
    if msa_spec:
        candidates.append(Path(msa_spec).expanduser())
    if msa_dir:
        h = hashlib.sha256(sequence.encode()).hexdigest()[:16]
        candidates.append(Path(msa_dir) / f"{h}.a3m")
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return MSA.from_a3m(str(p), max_sequences=max_sequences)
    return None


def fold_complex(model, chains, *, num_loops=3, num_sampling_steps=20,
                 num_diffusion_samples=1, seed=0):
    """Fold one (possibly multi-chain) protein complex on an already-patched model.

    `chains` is a list of ``(chain_id, sequence)`` or ``(chain_id, sequence,
    msa)`` where ``msa`` is an esm ``MSA`` object (or None for single-sequence).
    When an MSA is given the on-device MSA encoder runs. Returns the reference
    fold result (with `.complex`, `.plddt`, `.ptm`).

    With ``num_diffusion_samples > 1`` the diffusion head emits one structure per
    sample (distinct seeds); the reference ``fold`` returns them as a list. This
    is best-of-N folding, so we return the single highest-confidence sample,
    ranked by mean pLDDT (ESMFold's confidence metric) — not sample 0.
    """
    _ensure_reference_on_path()
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder, ProteinInput, StructurePredictionInput)

    def _protein(c):
        msa = c[2] if len(c) > 2 else None
        return ProteinInput(id=c[0], sequence=c[1], msa=msa)

    spi = StructurePredictionInput(sequences=[_protein(c) for c in chains])
    res = ESMFold2InputBuilder().fold(
        model, spi, num_loops=num_loops, num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples, seed=seed)
    if isinstance(res, list):
        return max(res, key=lambda r: float(r.plddt.mean()))
    return res
