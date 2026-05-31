"""Two calling-convention adapters around tt-boltz's ttnn modules.

Each adapter subclasses a tt-boltz ttnn module and only rewrites
``forward()`` to accept the calling convention BoltzGen's surrounding
code expects. They inherit ``TorchWrapper``'s state_dict mechanism so
production checkpoint weights flow in via the standard ``load_state_dict``.

Also exports :func:`load_boltz_checkpoint` — the inference entry point
that materialises a ``Boltz`` with all ttnn modules already constructed
(hard-wired in ``Boltz.__init__``) and loads weights, with a small remap
for the older ``token_transformer_layers.0.*`` key naming some shipping
design checkpoints still use.
"""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import sys
import types

import torch
import torch.nn as nn

from tt_boltz.tenstorrent import (
    DiffusionModule as TTDiffusionModule,
    PairformerModule as TTPairformerModule,
)


class _LegacyPickleStub:
    """Unpickle target for training-only classes referenced by old hparams.

    The shipping BoltzGen checkpoints were saved at training time and carry
    pickled references to validator dataclasses + ``torchmetrics`` loggers
    inside their ``hyper_parameters`` dict. None of those values are read at
    inference — ``load_boltz_checkpoint`` filters by ``Boltz.__init__``'s
    signature — but unpickling has to *find* the classes to construct
    placeholder instances and throw them away. This stub absorbs whatever
    state pickle hands it and returns ``None`` for any attribute access.
    """
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): self.__dict__.update(state if isinstance(state, dict) else {})
    def __getattr__(self, name): return None


@contextlib.contextmanager
def _legacy_pickle_compat():
    """Briefly satisfy ``torch.load`` for upstream checkpoints with deleted refs.

    Installs a ``sys.meta_path`` finder that fabricates placeholder modules for
    ``boltzgen.model.validation.*`` and ``torchmetrics.*`` so unpickling can
    find the (training-only) classes those names reference. Both the finder
    *and* the placeholder modules are removed on exit — leaving them in
    ``sys.modules`` past the load would pollute ``inspect``-based introspection
    (e.g. ``torch._library`` queries module ``__file__``).
    """
    installed = []
    finder = _LegacyModuleFinder(installed)
    sys.meta_path.append(finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for name in installed:
            sys.modules.pop(name, None)


class _LegacyModuleFinder:
    """``sys.meta_path`` finder fabricating placeholder modules on demand.

    Implements the modern ``find_spec`` / loader protocol. The legacy
    ``find_module`` / ``load_module`` protocol this used to rely on was removed
    from the import system in Python 3.12, so under 3.12 the old finder was
    silently never consulted and unpickling failed with
    ``ModuleNotFoundError: No module named 'boltzgen.model.validation'``.
    """

    PREFIXES = ("boltzgen.model.validation", "torchmetrics")

    def __init__(self, installed):
        self._installed = installed

    def find_spec(self, fullname, path=None, target=None):
        for p in self.PREFIXES:
            if fullname == p or fullname.startswith(p + "."):
                # is_package=True gives the stub a ``__path__`` so submodules
                # (e.g. ``torchmetrics.regression``) resolve through us too.
                return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        # Raise AttributeError on dunder names so introspection tools that
        # probe ``__file__``/``__spec__``/etc. don't receive the stub class
        # and try to call string methods on it.
        def _module_getattr(name, _stub=_LegacyPickleStub):
            if name.startswith("__"):
                raise AttributeError(name)
            return _stub
        m.__getattr__ = _module_getattr
        self._installed.append(spec.name)
        return m

    def exec_module(self, module):
        pass


class TTScoreModelAdapter(TTDiffusionModule):
    """Adapt the BoltzGen score-model calling convention to TTDiffusionModule.

    BoltzGen's ``AtomDiffusion.preconditioned_network_forward`` calls
    ``score_model(r_noisy=..., times=..., s_inputs=..., s_trunk=...,
    feats=..., diffusion_conditioning=..., multiplicity=...)`` and reads
    ``net_out["r_update"]``. ``TTDiffusionModule`` takes positional args
    and returns the raw tensor; this subclass shuffles kwargs into
    positional args, unwraps ``to_keys`` to its raw indexing matrix, and
    wraps the result in a dict.
    """

    def forward(  # type: ignore[override]
        self,
        *,
        r_noisy,
        times,
        s_inputs,
        s_trunk,
        feats,
        diffusion_conditioning,
        multiplicity: int = 1,
        **_unused,
    ):
        dc = diffusion_conditioning
        # ``to_keys`` is always ``partial(single_to_keys, indexing_matrix=K, W=32, H=128)``
        # built in encoders.py — unwrap the raw matrix.
        keys_indexing = dc["to_keys"].keywords["indexing_matrix"]
        r_update = super().forward(
            r_noisy, times, s_inputs, s_trunk,
            dc["q"], dc["c"],
            dc["atom_enc_bias"], dc["token_trans_bias"], dc["atom_dec_bias"],
            keys_indexing,
            feats["atom_pad_mask"], feats["atom_to_token"],
        )
        # BoltzGen reads only ["r_update"] from the design path; the other two
        # keys are unused (predict_res_type=False on every shipping ckpt).
        return {"r_update": r_update, "token_a": None, "res_type": None}


class TTPairformerNoSeqModule(TTPairformerModule):
    """No-seq Pairformer for BoltzGen's template / token-distance / affinity stacks.

    Same ttnn pairformer stack as the trunk, but ``transform_s=False`` (no
    s-track) and a forward signature that matches ``PairformerNoSeqModule.forward(z, pair_mask, use_kernels=...)``.
    """

    def __init__(self, n_blocks: int):
        super().__init__(
            n_blocks=n_blocks,
            tri_att_head_dim=32, tri_att_n_heads=4,
            att_head_dim=None, att_n_heads=None,
            transform_s=False,
        )

    def forward(self, z, pair_mask, use_kernels: bool = False):  # type: ignore[override]
        _, z_out = super().forward(None, z, mask=None, pair_mask=pair_mask)
        return z_out


def _remap_legacy_state_dict_keys(state: dict) -> dict:
    """Map older ``token_transformer_layers.0.*`` keys to ``token_transformer.*``.

    Mirrors what BoltzGen's old ``Boltz.on_load_checkpoint`` did. The shipping
    design checkpoint (``boltzgen1_diverse.ckpt``) predates the rename.
    """
    return {
        k.replace(".token_transformer_layers.0.", ".token_transformer."): v
        for k, v in state.items()
    }


def load_boltz_checkpoint(
    checkpoint_path: str,
    *,
    strict: bool = False,
    map_location: str = "cpu",
    **constructor_overrides,
) -> nn.Module:
    """Build a ``Boltz`` model from a checkpoint and load its weights.

    Filters legacy hyper_parameters the current Boltz signature no longer
    accepts, then remaps the legacy state-dict keys before loading.
    ``Boltz.__init__`` constructs ttnn modules directly — there is no
    PyTorch fallback path.
    """
    from tt_boltz.boltzgen.model.models.boltz import Boltz

    with _legacy_pickle_compat():
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False, mmap=True)
    sig = inspect.signature(Boltz.__init__).parameters
    hp = {k: v for k, v in ckpt["hyper_parameters"].items() if k in sig}
    hp.update({k: v for k, v in constructor_overrides.items() if k in sig})

    model = Boltz(**hp)
    state = _remap_legacy_state_dict_keys(ckpt["state_dict"])
    model.load_state_dict(state, strict=strict)
    return model.eval()
