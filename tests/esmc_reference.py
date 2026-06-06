"""Faithful PyTorch reference for ESMC, built from the real esm layers.

The upstream ``esm`` package (cloned at ``/home/ttuser/esm``) is not installed
and its top-level imports drag in heavy, irrelevant dependencies (zstd,
cloudpathlib, the SDK, tokenizer ...) via the *geometric attention* path that
ESMC never uses. We only need the plain transformer path, so we:

  1. put the esm clone on sys.path, and
  2. stub the two modules that solely back geometric attention
     (``esm.layers.geom_attention`` and ``esm.utils.structure.affine3d``),

then import the *real* RotaryEmbedding / MultiHeadAttention /
UnifiedTransformerBlock / TransformerStack / RegressionHead. ``ESMCReference``
below mirrors ``esm/models/esmc.py:123-191`` (with ``use_flash_attn=False``).

This is the golden reference our ttnn implementation is tested against. We load
identical weights into both and compare (the tt-boltz test idiom).
"""

from __future__ import annotations

import sys
import types

ESM_ROOT = "/home/ttuser/esm"
if ESM_ROOT not in sys.path:
    sys.path.insert(0, ESM_ROOT)

# --- stub the geometric-attention-only deps so the plain path imports cleanly ---
import torch.nn as nn  # noqa: E402


def _stub(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


# parent packages must exist before submodule stubs
import importlib  # noqa: E402

for pkg in ("esm", "esm.layers", "esm.utils", "esm.utils.structure"):
    importlib.import_module(pkg)

_stub("esm.layers.geom_attention", GeometricReasoningOriginalImpl=nn.Module)
_stub("esm.utils.structure.affine3d", Affine3D=object)

# --- real esm layers (plain transformer path) ---
import torch  # noqa: E402
from esm.layers.regression_head import RegressionHead  # noqa: E402
from esm.layers.transformer_stack import TransformerStack  # noqa: E402

# ---- ESMC-300M config (esm/pretrained.py:66-77) ----
ESMC_300M = dict(d_model=960, n_heads=15, n_layers=30)
VOCAB_SIZE = 64


class ESMCReference(nn.Module):
    """Mirrors esm.models.esmc.ESMC with use_flash_attn=False.

    forward(tokens) -> (sequence_logits[B,L,64], embeddings[B,L,d_model]).
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.transformer = TransformerStack(
            d_model, n_heads, None, n_layers, n_layers_geom=0, use_flash_attn=False
        )
        self.sequence_head = RegressionHead(d_model, VOCAB_SIZE)

    def forward(self, sequence_tokens: torch.Tensor, sequence_id=None):
        x = self.embed(sequence_tokens)
        x, _, _hidden, _attn = self.transformer(x, sequence_id)
        return self.sequence_head(x), x


def make_esmc_300m(seed: int = 0) -> ESMCReference:
    torch.manual_seed(seed)
    return ESMCReference(**ESMC_300M).eval()
