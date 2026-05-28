"""Tenstorrent-only BoltzGen smoke tests.

The earlier random-weight comparison tests are gone — they compared
tt-boltz's ttnn modules against PyTorch reference modules in
``boltzgen/model/layers/{pairformer,miniformer,...}.py``, all of which
have been deleted as part of the Tenstorrent-only strip. The model now
constructs ttnn modules directly in ``Boltz.__init__`` and the PyTorch
fallback path doesn't exist.

What's left:
  * ``test_design_checkpoint_loads``: build ``Boltz`` from the production
    design checkpoint and confirm the state dict loads with zero missing
    / zero unexpected keys.
  * ``test_folding_checkpoint_loads``: same for the folding checkpoint.

These prove the seam (state dict layout) hasn't drifted, which is the
narrowest failure mode any future refactor can introduce.
"""
import os
from pathlib import Path

import pytest


_CKPT_ROOT = (
    Path.home()
    / ".cache/huggingface/hub/models--boltzgen--boltzgen-1"
    / "snapshots/c1be29e1f82ffcc72264f64b993c43fb4e0d17f0"
)
DESIGN_CKPT = Path(os.environ.get("BOLTZGEN_DESIGN_CKPT", _CKPT_ROOT / "boltzgen1_diverse.ckpt"))
FOLD_CKPT = Path(os.environ.get("BOLTZGEN_FOLD_CKPT", _CKPT_ROOT / "boltz2_conf_final.ckpt"))


def _load(ckpt: Path) -> int:
    """Build Boltz from ``ckpt`` and load weights; return unexpected-key count."""
    import inspect
    import torch
    from tt_boltz.boltzgen.adapter import _remap_legacy_state_dict_keys
    from tt_boltz.boltzgen.model.models.boltz import Boltz

    blob = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    sig = inspect.signature(Boltz.__init__).parameters
    hp = {k: v for k, v in blob["hyper_parameters"].items() if k in sig}
    model = Boltz(**hp).eval()
    state = _remap_legacy_state_dict_keys(blob["state_dict"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, f"missing keys after load: {missing[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    return 0


@pytest.mark.skipif(not DESIGN_CKPT.exists(), reason=f"missing {DESIGN_CKPT}")
def test_design_checkpoint_loads() -> None:
    _load(DESIGN_CKPT)


@pytest.mark.skipif(not FOLD_CKPT.exists(), reason=f"missing {FOLD_CKPT}")
def test_folding_checkpoint_loads() -> None:
    _load(FOLD_CKPT)
