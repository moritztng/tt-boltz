"""BoltzGen vendored into tt-boltz, stripped to Tenstorrent-only inference.

User-facing entry points live in :mod:`tt_boltz.boltzgen.cli` (CLI) and
:mod:`tt_boltz.boltzgen.adapter` (the ttnn-module adapters + checkpoint
loader). Everything else is the vendored BoltzGen source, with training-only
code (validators, optimizers, training-time losses, training data filters/
samplers) removed and PyTorch Lightning + Hydra dependencies eliminated.

The shipping BoltzGen checkpoints pickle ``hyper_parameters`` that reference
``boltzgen.X`` classes. Pickle imports ``boltzgen`` before resolving anything
under it, so we keep a bare module alias here. Resolution of deleted
*training-only* class names is scoped to ``adapter._legacy_pickle_compat``.
"""
import sys as _sys

_sys.modules.setdefault("boltzgen", _sys.modules[__name__])

from tt_boltz.boltzgen.adapter import (  # noqa: E402
    load_boltz_checkpoint,
    TTPairformerNoSeqModule,
    TTScoreModelAdapter,
)

__all__ = [
    "load_boltz_checkpoint",
    "TTPairformerNoSeqModule",
    "TTScoreModelAdapter",
]
