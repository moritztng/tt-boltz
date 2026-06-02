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
