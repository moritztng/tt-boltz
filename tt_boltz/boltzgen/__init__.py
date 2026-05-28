"""BoltzGen vendored into tt-boltz, stripped to Tenstorrent-only inference.

User-facing entry points live in :mod:`tt_boltz.boltzgen.cli` (CLI) and
:mod:`tt_boltz.boltzgen.adapter` (the ``convert_to_tt`` swap + checkpoint
loader). Everything else is the vendored BoltzGen source, with training-only
code (validators, optimizers, training-time loss-step uses, training data
filters/samplers) removed and PyTorch Lightning + Hydra dependencies eliminated.

The shipping BoltzGen checkpoints pickle hyper_parameters that reference
``boltzgen.X`` classes (training_args dataclasses, validator configs, etc.).
We never use any of those at inference time — ``adapter.load_boltz_checkpoint``
filters them out by ``inspect.signature(Boltz.__init__)`` — but unpickling
still has to *find* them. We register ``boltzgen`` as an alias for
``tt_boltz.boltzgen`` in ``sys.modules`` so pickle resolves the class refs
to our vendored source (or, if the class is purely training-only and now
deleted, to a placeholder that unpickles to ``None``).
"""
import sys as _sys
import types as _types

# Make the upstream boltzgen.X paths resolve to our vendored copies for the
# purpose of unpickling checkpoint hparams. Loaders never actually call these
# objects — the surviving inference path uses tt_boltz.boltzgen.X explicitly.
_sys.modules.setdefault("boltzgen", _sys.modules[__name__])


class _DeletedTrainingArtifact:
    """Placeholder for training-only classes referenced by old checkpoint hparams.

    Returning instances of this from unpickling lets ``torch.load`` complete
    even though the class itself no longer exists in our tree (validators,
    EMA configs, training_args dataclasses). Inference filters these out
    before they reach any model code.
    """
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})
    def __getattr__(self, name):
        return None


def _missing_module_factory(fullname: str) -> _types.ModuleType:
    """Return a fake package whose every attribute is :class:`_DeletedTrainingArtifact`."""
    m = _types.ModuleType(fullname)
    m.__path__ = []   # mark as package so submodule imports proceed via our finder
    def __getattr__(name):
        return _DeletedTrainingArtifact
    m.__getattr__ = __getattr__
    return m


class _DeletedModuleFinder:
    """``sys.meta_path`` finder that fabricates fake modules for deleted
    training paths so old checkpoints can still be unpickled."""

    _DELETED_PREFIXES = (
        "boltzgen.task.train",
        "boltzgen.model.optim",
        "boltzgen.model.validation",
        "boltzgen.data.filter",
        "boltzgen.data.sample",
        "tt_boltz.boltzgen.task.train",
        "tt_boltz.boltzgen.model.optim",
        "tt_boltz.boltzgen.model.validation",
        "tt_boltz.boltzgen.data.filter",
        "tt_boltz.boltzgen.data.sample",
        # External packages the shipping checkpoints pickle into hyper_parameters
        # for training-time concepts we no longer carry. We never touch these
        # objects at inference — they're just there to satisfy the unpickler.
        "torchmetrics",
        "pytorch_lightning",
    )

    def find_module(self, fullname, path=None):
        for prefix in self._DELETED_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return self
        return None

    def load_module(self, fullname):
        if fullname in _sys.modules:
            return _sys.modules[fullname]
        m = _missing_module_factory(fullname)
        _sys.modules[fullname] = m
        return m


_sys.meta_path.append(_DeletedModuleFinder())

from tt_boltz.boltzgen.adapter import (
    load_boltz_checkpoint,
    TTPairformerNoSeqModule,
    TTScoreModelAdapter,
)

__all__ = [
    "load_boltz_checkpoint",
    "TTPairformerNoSeqModule",
    "TTScoreModelAdapter",
]
