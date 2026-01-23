# Boltz-2: Unified PyTorch model file
# This file combines utilities, potentials, core modules, model components, and Boltz2 class.
# For Tenstorrent-specific modules, see tenstorrent.py
# For CPU/reference fallback modules, see reference.py

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from functools import partial
from math import exp, pi, sqrt
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from scipy.stats import truncnorm
from torch import nn, Tensor
from torch.nn import Linear as TorchLinear, Module, ModuleList, Sequential
from torch.nn.functional import one_hot, pad

from tt_boltz import tenstorrent
from tt_boltz.data import const

# Lazy imports for fallback modules to avoid circular imports
# These are imported inside classes that use them
def _get_pytorch_modules():
    from tt_boltz.reference import (
        DiffusionModule,
    PairformerModule,
        PairformerNoSeqModule,
        MSAModule,
    )
    return DiffusionModule, PairformerModule, PairformerNoSeqModule, MSAModule


# ============================================================================
# BASIC UTILITIES (no dependencies on other sections)
# ============================================================================

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape
    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")
    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    torch.nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def bias_init_zero_(bias):
    with torch.no_grad():
        bias.fill_(0.0)


def bias_init_one_(bias):
    with torch.no_grad():
        bias.fill_(1.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


# Namespace for initialization functions (used throughout the codebase)
init = SimpleNamespace(
    lecun_normal_init_=lecun_normal_init_,
    he_normal_init_=he_normal_init_,
    glorot_uniform_init_=glorot_uniform_init_,
    trunc_normal_init_=trunc_normal_init_,
    normal_init_=normal_init_,
    final_init_=final_init_,
    gating_init_=gating_init_,
    bias_init_zero_=bias_init_zero_,
    bias_init_one_=bias_init_one_,
)
initialize = init  # Alias for backward compatibility


# ============================================================================
# RANDOM AUGMENTATION UTILITIES
# ============================================================================

def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_quaternions(n: int, dtype=None, device=None) -> torch.Tensor:
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(n: int, dtype=None, device=None) -> torch.Tensor:
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)


def compute_random_augmentation(multiplicity, s_trans=1.0, device=None, dtype=torch.float32):
    R = random_rotations(multiplicity, dtype=dtype, device=device)
    random_trans = torch.randn((multiplicity, 1, 3), dtype=dtype, device=device) * s_trans
    return R, random_trans


def randomly_rotate(coords, return_second_coords=False, second_coords=None):
    R = random_rotations(len(coords), coords.dtype, coords.device)
    if return_second_coords:
        return torch.einsum("bmd,bds->bms", coords, R), torch.einsum(
            "bmd,bds->bms", second_coords, R
        ) if second_coords is not None else None
    return torch.einsum("bmd,bds->bms", coords, R)


def center_random_augmentation(
    atom_coords,
    atom_mask,
    s_trans=1.0,
    augmentation=True,
    centering=True,
    return_second_coords=False,
    second_coords=None,
):
    """Algorithm 19"""
    if centering:
        atom_mean = torch.sum(
            atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
        ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
        atom_coords = atom_coords - atom_mean
        if second_coords is not None:
            second_coords = second_coords - atom_mean

    if augmentation:
        atom_coords, second_coords = randomly_rotate(
            atom_coords, return_second_coords=True, second_coords=second_coords
        )
        random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * s_trans
        atom_coords = atom_coords + random_trans
        if second_coords is not None:
            second_coords = second_coords + random_trans

    if return_second_coords:
        return atom_coords, second_coords
    return atom_coords


# ============================================================================
# ALIGNMENT UTILITIES
# ============================================================================

def weighted_rigid_align(true_coords, pred_coords, weights, mask):
    out_shape = torch.broadcast_shapes(true_coords.shape, pred_coords.shape)
    *batch_size, num_points, dim = out_shape
    weights = (mask * weights).unsqueeze(-1)

    true_centroid = (true_coords * weights).sum(dim=-2, keepdim=True) / weights.sum(dim=-2, keepdim=True)
    pred_centroid = (pred_coords * weights).sum(dim=-2, keepdim=True) / weights.sum(dim=-2, keepdim=True)
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if torch.any(mask.sum(dim=-1) < (dim + 1)):
        print("Warning: The size of one of the point clouds is <= dim+1.")

    cov_matrix = torch.einsum(
        "... n i, ... n j -> ... i j", weights * pred_coords_centered, true_coords_centered
    )

    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    U, S, V = torch.linalg.svd(cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None)
    V = V.mH

    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print("Warning: Excessively low rank of cross-correlation.")

    rot_matrix = torch.einsum("... i j, ... k j -> ... i k", U, V).to(dtype=torch.float32)

    F_mat = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device)[None].repeat(*batch_size, 1, 1)
    F_mat[..., -1, -1] = torch.det(rot_matrix)
    rot_matrix = torch.einsum("... i j, ... j k, ... l k -> ... i l", U, F_mat, V)
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    aligned_coords = (
        torch.einsum("... n i, ... j i -> ... n j", true_coords_centered, rot_matrix)
        + pred_centroid
    )
    return aligned_coords


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

class SwiGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


# ============================================================================
# DROPOUT AND ATTENTION UTILITIES
# ============================================================================

def get_dropout_mask(
    dropout: float,
    z: Tensor,
    training: bool,
    columnwise: bool = False,
) -> Tensor:
    """Get the dropout mask."""
    dropout = dropout * training
    v = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    d = torch.rand(v.shape, dtype=torch.float32, device=v.device) >= dropout
    d = d * 1.0 / (1.0 - dropout)
    return d


class AttentionPairBias(nn.Module):
    """Attention pair bias layer."""

    def __init__(
        self,
        c_s: int,
        c_z: Optional[int] = None,
        num_heads: Optional[int] = None,
        inf: float = 1e6,
        compute_pair_bias: bool = True,
    ) -> None:
        super().__init__()
        assert c_s % num_heads == 0
        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.compute_pair_bias = compute_pair_bias
        if compute_pair_bias:
            self.proj_z = nn.Sequential(
                nn.LayerNorm(c_z),
                nn.Linear(c_z, num_heads, bias=False),
                Rearrange("b ... h -> b h ..."),
            )
        else:
            self.proj_z = Rearrange("b ... h -> b h ...")

        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        init.final_init_(self.proj_o.weight)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        k_in: Tensor,
        multiplicity: int = 1,
    ) -> Tensor:
        B = s.shape[0]
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)
        bias = self.proj_z(z)
        bias = bias.repeat_interleave(multiplicity, 0)
        g = self.proj_g(s).sigmoid()

        with torch.autocast("cuda", enabled=False):
            attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            attn = attn / (self.head_dim**0.5) + bias.float()
            attn = attn + (1 - mask[:, None, None].float()) * -self.inf
            attn = attn.softmax(dim=-1)
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)
        return o


# ============================================================================
# TENSOR TREE MAP AND CHUNK UTILITIES
# ============================================================================

def add(m1, m2, inplace):
    if not inplace:
        m1 = m1 + m2
    else:
        m1 += m2
    return m1


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)
    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        raise ValueError(f"Tree of type {type(tree)} not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def _fetch_dims(tree):
    shapes = []
    tree_type = type(tree)
    if tree_type is dict:
        for v in tree.values():
            shapes.extend(_fetch_dims(v))
    elif tree_type is list or tree_type is tuple:
        for t in tree:
            shapes.extend(_fetch_dims(t))
    elif tree_type is torch.Tensor:
        shapes.append(tree.shape)
    else:
        raise ValueError("Not supported")
    return shapes


@torch.jit.ignore
def _flat_idx_to_idx(flat_idx: int, dims: Tuple[int]) -> Tuple[int]:
    idx = []
    for d in reversed(dims):
        idx.append(flat_idx % d)
        flat_idx = flat_idx // d
    return tuple(reversed(idx))


@torch.jit.ignore
def _get_minimal_slice_set(
    start: Sequence[int],
    end: Sequence[int],
    dims: int,
    start_edges: Optional[Sequence[bool]] = None,
    end_edges: Optional[Sequence[bool]] = None,
) -> Sequence[Tuple[int]]:
    def reduce_edge_list(l):
        tally = 1
        for i in range(len(l)):
            reversed_idx = -1 * (i + 1)
            l[reversed_idx] *= tally
            tally = l[reversed_idx]

    if start_edges is None:
        start_edges = [s == 0 for s in start]
        reduce_edge_list(start_edges)
    if end_edges is None:
        end_edges = [e == (d - 1) for e, d in zip(end, dims)]
        reduce_edge_list(end_edges)

    if len(start) == 0:
        return [tuple()]
    elif len(start) == 1:
        return [(slice(start[0], end[0] + 1),)]

    slices = []
    path = []
    for s, e in zip(start, end):
        if s == e:
            path.append(slice(s, s + 1))
        else:
            break
    path = tuple(path)
    divergence_idx = len(path)

    if divergence_idx == len(dims):
        return [tuple(path)]

    def upper():
        sdi = start[divergence_idx]
        return [
            path + (slice(sdi, sdi + 1),) + s
            for s in _get_minimal_slice_set(
                start[divergence_idx + 1 :],
                [d - 1 for d in dims[divergence_idx + 1 :]],
                dims[divergence_idx + 1 :],
                start_edges=start_edges[divergence_idx + 1 :],
                end_edges=[1 for _ in end_edges[divergence_idx + 1 :]],
            )
        ]

    def lower():
        edi = end[divergence_idx]
        return [
            path + (slice(edi, edi + 1),) + s
            for s in _get_minimal_slice_set(
                [0 for _ in start[divergence_idx + 1 :]],
                end[divergence_idx + 1 :],
                dims[divergence_idx + 1 :],
                start_edges=[1 for _ in start_edges[divergence_idx + 1 :]],
                end_edges=end_edges[divergence_idx + 1 :],
            )
        ]

    if start_edges[divergence_idx] and end_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx] + 1),))
    elif start_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx]),))
        slices.extend(lower())
    elif end_edges[divergence_idx]:
        slices.extend(upper())
        slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx] + 1),))
    else:
        slices.extend(upper())
        middle_ground = end[divergence_idx] - start[divergence_idx]
        if middle_ground > 1:
            slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx]),))
        slices.extend(lower())

    return [tuple(s) for s in slices]


@torch.jit.ignore
def _chunk_slice(
    t: torch.Tensor,
    flat_start: int,
    flat_end: int,
    no_batch_dims: int,
) -> torch.Tensor:
    batch_dims = t.shape[:no_batch_dims]
    start_idx = list(_flat_idx_to_idx(flat_start, batch_dims))
    end_idx = list(_flat_idx_to_idx(flat_end - 1, batch_dims))
    slices = _get_minimal_slice_set(start_idx, end_idx, batch_dims)
    sliced_tensors = [t[s] for s in slices]
    return torch.cat([s.view((-1,) + t.shape[no_batch_dims:]) for s in sliced_tensors])


def chunk_layer(
    layer: Callable,
    inputs: Dict[str, Any],
    chunk_size: int,
    no_batch_dims: int,
    low_mem: bool = False,
    _out: Any = None,
    _add_into_out: bool = False,
) -> Any:
    if not (len(inputs) > 0):
        raise ValueError("Must provide at least one input")

    initial_dims = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
    orig_batch_dims = tuple([max(s) for s in zip(*initial_dims)])

    def _prep_inputs(t):
        if not low_mem:
            if not sum(t.shape[:no_batch_dims]) == no_batch_dims:
                t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
            t = t.reshape(-1, *t.shape[no_batch_dims:])
        else:
            t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
        return t

    prepped_inputs = tensor_tree_map(_prep_inputs, inputs)
    prepped_outputs = None
    if _out is not None:
        reshape_fn = lambda t: t.view([-1] + list(t.shape[no_batch_dims:]))
        prepped_outputs = tensor_tree_map(reshape_fn, _out)

    flat_batch_dim = 1
    for d in orig_batch_dims:
        flat_batch_dim *= d

    no_chunks = flat_batch_dim // chunk_size + (flat_batch_dim % chunk_size != 0)

    i = 0
    out = prepped_outputs
    for _ in range(no_chunks):
        if not low_mem:
            select_chunk = lambda t: t[i : i + chunk_size] if t.shape[0] != 1 else t
        else:
            select_chunk = partial(
                _chunk_slice,
                flat_start=i,
                flat_end=min(flat_batch_dim, i + chunk_size),
                no_batch_dims=len(orig_batch_dims),
            )

        chunks = tensor_tree_map(select_chunk, prepped_inputs)
        output_chunk = layer(**chunks)

        if out is None:
            allocate = lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:])
            out = tensor_tree_map(allocate, output_chunk)

        out_type = type(output_chunk)
        if out_type is dict:
            def assign(d1, d2):
                for k, v in d1.items():
                    if type(v) is dict:
                        assign(v, d2[k])
                    else:
                        if _add_into_out:
                            v[i : i + chunk_size] += d2[k]
                        else:
                            v[i : i + chunk_size] = d2[k]
            assign(out, output_chunk)
        elif out_type is tuple:
            for x1, x2 in zip(out, output_chunk):
                if _add_into_out:
                    x1[i : i + chunk_size] += x2
                else:
                    x1[i : i + chunk_size] = x2
        elif out_type is torch.Tensor:
            if _add_into_out:
                out[i : i + chunk_size] += output_chunk
            else:
                out[i : i + chunk_size] = output_chunk
        else:
            raise ValueError("Not supported")

        i += chunk_size

    reshape = lambda t: t.view(orig_batch_dims + t.shape[1:])
    out = tensor_tree_map(reshape, out)
    return out


# ============================================================================
# LINEAR & LAYER NORM WITH CUSTOM INITIALIZATION
# ============================================================================

LinearNoBias = partial(TorchLinear, bias=False)


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
        precision=None,
    ):
        """Initialize the linear layer.

        Parameters
        ----------
        in_dim : int
            The final dimension of inputs to the layer
        out_dim : int
            The final dimension of layer outputs
        bias : bool, default=True
            Whether to learn an additive bias
        init : str, default='default'
            The initializer to use. Choose from:

            - "default": LeCun fan-in truncated normal initialization
            - "relu": He initialization w/ truncated normal distribution
            - "glorot": Fan-average Glorot uniform initialization
            - "gating": Weights=0, Bias=1
            - "normal": Normal initialization with std=1/sqrt(fan_in)
            - "final": Weights=0, Bias=0

            Overridden by init_fn if the latter is not None.
        init_fn : callable, optional
            A custom initializer taking weight and bias as inputs.
            Overrides init if not None.

        """
        super().__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    initialize.lecun_normal_init_(self.weight)
                elif init == "relu":
                    initialize.he_normal_init_(self.weight)
                elif init == "glorot":
                    initialize.glorot_uniform_init_(self.weight)
                elif init == "gating":
                    initialize.gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    initialize.normal_init_(self.weight)
                elif init == "final":
                    initialize.final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")

        self.precision = precision

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        d = input.dtype
        if self.precision is not None:
            with torch.autocast("cuda", enabled=False):
                bias = (
                    self.bias.to(dtype=self.precision)
                    if self.bias is not None
                    else None
                )
                return nn.functional.linear(
                    input.to(dtype=self.precision),
                    self.weight.to(dtype=self.precision),
                    bias,
                ).to(dtype=d)

        if d is torch.bfloat16:
            with torch.autocast("cuda", enabled=False):
                bias = self.bias.to(dtype=d) if self.bias is not None else None
                return nn.functional.linear(input, self.weight.to(dtype=d), bias)

        return nn.functional.linear(input, self.weight, self.bias)


# Preserve custom Linear to restore after later imports override it.
LinearWithInit = Linear


class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        d = x.dtype
        if d is torch.bfloat16:
            with torch.autocast("cuda", enabled=False):
                out = nn.functional.layer_norm(
                    x,
                    self.c_in,
                    self.weight.to(dtype=d),
                    self.bias.to(dtype=d),
                    self.eps,
                )
        else:
            out = nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )

        return out


@torch.jit.ignore

def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of
    type bfloat16
    """
    d = t.dtype
    if d is torch.bfloat16:
        with torch.autocast("cuda", enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


# @torch.jit.script

def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: List[torch.Tensor],
) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


@torch.compiler.disable

def kernel_triangular_attn(q, k, v, tri_bias, mask, scale):
    from cuequivariance_torch.primitives.triangle import triangle_attention
    return triangle_attention(q, k, v, tri_bias, mask=mask, scale=scale)


def kernel_triangular_mult(
    x,
    direction,
    mask,
    norm_in_weight,
    norm_in_bias,
    p_in_weight,
    g_in_weight,
    norm_out_weight,
    norm_out_bias,
    p_out_weight,
    g_out_weight,
    eps,
):
    from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
    return triangle_multiplicative_update(
        x,
        direction=direction,
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=eps,
    )


def compute_collinear_mask(v1, v2):
    norm1 = torch.norm(v1, dim=1, keepdim=True)
    norm2 = torch.norm(v2, dim=1, keepdim=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = torch.abs(torch.sum(v1 * v2, dim=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def compute_frame_pred(
    pred_atom_coords,
    frames_idx_true,
    feats,
    multiplicity,
    resolved_mask=None,
    inference=False,
):
    with torch.amp.autocast("cuda", enabled=False):
        asym_id_token = feats["asym_id"]
        asym_id_atom = torch.bmm(
            feats["atom_to_token"].float(), asym_id_token.unsqueeze(-1).float()
        ).squeeze(-1)

    B, N, _ = pred_atom_coords.shape
    pred_atom_coords = pred_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    frames_idx_pred = (
        frames_idx_true.clone()
        .repeat_interleave(multiplicity, 0)
        .reshape(B // multiplicity, multiplicity, -1, 3)
    )

    # Iterate through the batch and modify the frames for nonpolymers
    for i, pred_atom_coord in enumerate(pred_atom_coords):
        token_idx = 0
        atom_idx = 0
        for id in torch.unique(asym_id_token[i]):
            mask_chain_token = (asym_id_token[i] == id) * feats["token_pad_mask"][i]
            mask_chain_atom = (asym_id_atom[i] == id) * feats["atom_pad_mask"][i]
            num_tokens = int(mask_chain_token.sum().item())
            num_atoms = int(mask_chain_atom.sum().item())
            if (
                feats["mol_type"][i, token_idx] != const.chain_type_ids["NONPOLYMER"]
                or num_atoms < 3
            ):
                token_idx += num_tokens
                atom_idx += num_atoms
                continue
            dist_mat = (
                (
                    pred_atom_coord[:, mask_chain_atom.bool()][:, None, :, :]
                    - pred_atom_coord[:, mask_chain_atom.bool()][:, :, None, :]
                )
                ** 2
            ).sum(-1) ** 0.5
            if inference:
                resolved_pair = 1 - (
                    feats["atom_pad_mask"][i][mask_chain_atom.bool()][None, :]
                    * feats["atom_pad_mask"][i][mask_chain_atom.bool()][:, None]
                ).to(torch.float32)
                resolved_pair[resolved_pair == 1] = torch.inf
                indices = torch.sort(dist_mat + resolved_pair, axis=2).indices
            else:
                if resolved_mask is None:
                    resolved_mask = feats["atom_resolved_mask"]
                resolved_pair = 1 - (
                    resolved_mask[i][mask_chain_atom.bool()][None, :]
                    * resolved_mask[i][mask_chain_atom.bool()][:, None]
                ).to(torch.float32)
                resolved_pair[resolved_pair == 1] = torch.inf
                indices = torch.sort(dist_mat + resolved_pair, axis=2).indices
            frames = (
                torch.cat(
                    [
                        indices[:, :, 1:2],
                        indices[:, :, 0:1],
                        indices[:, :, 2:3],
                    ],
                    dim=2,
                )
                + atom_idx
            )
            try:
                frames_idx_pred[i, :, token_idx : token_idx + num_atoms, :] = frames
            except Exception as e:
                print(f"Failed to process {feats['pdb_id']} due to {e}")
            token_idx += num_tokens
            atom_idx += num_atoms

    frames_expanded = pred_atom_coords[
        torch.arange(0, B // multiplicity, 1)[:, None, None, None].to(
            frames_idx_pred.device
        ),
        torch.arange(0, multiplicity, 1)[None, :, None, None].to(
            frames_idx_pred.device
        ),
        frames_idx_pred,
    ].reshape(-1, 3, 3)

    # Compute masks for collinearity / overlap
    mask_collinear_pred = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    ).reshape(B // multiplicity, multiplicity, -1)
    return frames_idx_pred, mask_collinear_pred * feats["token_pad_mask"][:, None, :]


def compute_aggregated_metric(logits, end=1.0):
    # Compute aggregated metric from logits
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=end, step=bin_width, device=logits.device
    )
    probs = nn.functional.softmax(logits, dim=-1)
    plddt = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return plddt


def tm_function(d, Nres):
    d0 = 1.24 * (torch.clip(Nres, min=19) - 15) ** (1 / 3) - 1.8
    return 1 / (1 + (d / d0) ** 2)


def compute_ptms(logits, x_preds, feats, multiplicity):
    # It needs to take as input the mask of the frames as they are not used to compute the PTM
    _, mask_collinear_pred = compute_frame_pred(
        x_preds, feats["frames_idx"], feats, multiplicity, inference=True
    )
    # mask overlapping, collinear tokens and ions (invalid frames)
    mask_pad = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
    maski = mask_collinear_pred.reshape(-1, mask_collinear_pred.shape[-1])
    pair_mask_ptm = maski[:, :, None] * mask_pad[:, None, :] * mask_pad[:, :, None]
    asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    pair_mask_iptm = (
        maski[:, :, None]
        * (asym_id[:, None, :] != asym_id[:, :, None])
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
    )
    num_bins = logits.shape[-1]
    bin_width = 32.0 / num_bins
    end = 32.0
    pae_value = torch.arange(
        start=0.5 * bin_width, end=end, step=bin_width, device=logits.device
    ).unsqueeze(0)
    N_res = mask_pad.sum(dim=-1, keepdim=True)
    tm_value = tm_function(pae_value, N_res).unsqueeze(1).unsqueeze(2)
    probs = nn.functional.softmax(logits, dim=-1)
    tm_expected_value = torch.sum(
        probs * tm_value,
        dim=-1,
    )  # shape (B, N, N)
    ptm = torch.max(
        torch.sum(tm_expected_value * pair_mask_ptm, dim=-1)
        / (torch.sum(pair_mask_ptm, dim=-1) + 1e-5),
        dim=1,
    ).values
    iptm = torch.max(
        torch.sum(tm_expected_value * pair_mask_iptm, dim=-1)
        / (torch.sum(pair_mask_iptm, dim=-1) + 1e-5),
        dim=1,
    ).values

    # compute ligand and protein iPTM
    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)
    is_ligand_token = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    is_protein_token = (token_type == const.chain_type_ids["PROTEIN"]).float()

    ligand_iptm_mask = (
        maski[:, :, None]
        * (asym_id[:, None, :] != asym_id[:, :, None])
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
        * (
            (is_ligand_token[:, :, None] * is_protein_token[:, None, :])
            + (is_protein_token[:, :, None] * is_ligand_token[:, None, :])
        )
    )
    protein_ipmt_mask = (
        maski[:, :, None]
        * (asym_id[:, None, :] != asym_id[:, :, None])
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
        * (is_protein_token[:, :, None] * is_protein_token[:, None, :])
    )

    ligand_iptm = torch.max(
        torch.sum(tm_expected_value * ligand_iptm_mask, dim=-1)
        / (torch.sum(ligand_iptm_mask, dim=-1) + 1e-5),
        dim=1,
    ).values
    protein_iptm = torch.max(
        torch.sum(tm_expected_value * protein_ipmt_mask, dim=-1)
        / (torch.sum(protein_ipmt_mask, dim=-1) + 1e-5),
        dim=1,
    ).values

    # Compute pair chain ipTM
    chain_pair_iptm = {}
    asym_ids_list = torch.unique(asym_id).tolist()
    for idx1 in asym_ids_list:
        chain_iptm = {}
        for idx2 in asym_ids_list:
            mask_pair_chain = (
                maski[:, :, None]
                * (asym_id[:, None, :] == idx1)
                * (asym_id[:, :, None] == idx2)
                * mask_pad[:, None, :]
                * mask_pad[:, :, None]
            )

            chain_iptm[idx2] = torch.max(
                torch.sum(tm_expected_value * mask_pair_chain, dim=-1)
                / (torch.sum(mask_pair_chain, dim=-1) + 1e-5),
                dim=1,
            ).values
        chain_pair_iptm[idx1] = chain_iptm

    return ptm, iptm, ligand_iptm, protein_iptm, chain_pair_iptm

# ---- encodersv2.py ----

# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang


class FourierEmbedding(Module):
    """Algorithm 22."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        torch.nn.init.normal_(self.proj.weight, mean=0, std=1)
        torch.nn.init.normal_(self.proj.bias, mean=0, std=1)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times,  # Float[' b'],
    ):  # -> Float['b d']:
        times = rearrange(times, "b -> b 1")
        rand_proj = self.proj(times)
        return torch.cos(2 * pi * rand_proj)


class RelativePositionEncoder(Module):
    """Algorithm 3."""

    def __init__(
        self, token_z, r_max=32, s_max=2, fix_sym_check=False, cyclic_pos_enc=False
    ):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.linear_layer = LinearNoBias(4 * (r_max + 1) + 2 * (s_max + 1) + 1, token_z)
        self.fix_sym_check = fix_sym_check
        self.cyclic_pos_enc = cyclic_pos_enc

    def forward(self, feats):
        b_same_chain = torch.eq(
            feats["asym_id"][:, :, None], feats["asym_id"][:, None, :]
        )
        b_same_residue = torch.eq(
            feats["residue_index"][:, :, None], feats["residue_index"][:, None, :]
        )
        b_same_entity = torch.eq(
            feats["entity_id"][:, :, None], feats["entity_id"][:, None, :]
        )

        d_residue = (
            feats["residue_index"][:, :, None] - feats["residue_index"][:, None, :]
        )

        if self.cyclic_pos_enc and torch.any(feats["cyclic_period"] > 0):
            period = torch.where(
                feats["cyclic_period"] > 0,
                feats["cyclic_period"],
                torch.zeros_like(feats["cyclic_period"]) + 10000,
            )
            d_residue = (d_residue - period * torch.round(d_residue / period)).long()

        d_residue = torch.clip(
            d_residue + self.r_max,
            0,
            2 * self.r_max,
        )
        d_residue = torch.where(
            b_same_chain, d_residue, torch.zeros_like(d_residue) + 2 * self.r_max + 1
        )
        a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)

        d_token = torch.clip(
            feats["token_index"][:, :, None]
            - feats["token_index"][:, None, :]
            + self.r_max,
            0,
            2 * self.r_max,
        )
        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            torch.zeros_like(d_token) + 2 * self.r_max + 1,
        )
        a_rel_token = one_hot(d_token, 2 * self.r_max + 2)

        d_chain = torch.clip(
            feats["sym_id"][:, :, None] - feats["sym_id"][:, None, :] + self.s_max,
            0,
            2 * self.s_max,
        )
        d_chain = torch.where(
            (~b_same_entity) if self.fix_sym_check else b_same_chain,
            torch.zeros_like(d_chain) + 2 * self.s_max + 1,
            d_chain,
        )
        # Note: added  | (~b_same_entity) based on observation of ProteinX manuscript
        a_rel_chain = one_hot(d_chain, 2 * self.s_max + 2)

        p = self.linear_layer(
            torch.cat(
                [
                    a_rel_pos.float(),
                    a_rel_token.float(),
                    b_same_entity.unsqueeze(-1).float(),
                    a_rel_chain.float(),
                ],
                dim=-1,
            )
        )
        return p


class Transition(nn.Module):
    """Perform a two-layer MLP."""

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        out_dim: Optional[int] = None,
    ) -> None:
        """Initialize the TransitionUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        hidden: int
            The dimension of the hidden, default 512
        out_dim: Optional[int]
            The dimension of the output, default None

        """
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.silu = nn.SiLU()
        self.hidden = hidden

        init.bias_init_one_(self.norm.weight)
        init.bias_init_zero_(self.norm.bias)

        init.lecun_normal_init_(self.fc1.weight)
        init.lecun_normal_init_(self.fc2.weight)
        init.final_init_(self.fc3.weight)

    def forward(self, x: Tensor, chunk_size: int = None) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (..., D)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (..., D)

        """
        x = self.norm(x)

        if chunk_size is None or self.training:
            x = self.silu(self.fc1(x)) * self.fc2(x)
            x = self.fc3(x)
            return x
        else:
            # Compute in chunks
            for i in range(0, self.hidden, chunk_size):
                fc1_slice = self.fc1.weight[i : i + chunk_size, :]
                fc2_slice = self.fc2.weight[i : i + chunk_size, :]
                fc3_slice = self.fc3.weight[:, i : i + chunk_size]
                x_chunk = self.silu((x @ fc1_slice.T)) * (x @ fc2_slice.T)
                if i == 0:
                    x_out = x_chunk @ fc3_slice.T
                else:
                    x_out = x_out + x_chunk @ fc3_slice.T
            return x_out


class SingleConditioning(Module):
    """Algorithm 21."""

    def __init__(
        self,
        sigma_data: float,
        token_s: int = 384,
        dim_fourier: int = 256,
        num_transitions: int = 2,
        transition_expansion_factor: int = 2,
        eps: float = 1e-20,
        disable_times: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.sigma_data = sigma_data
        self.disable_times = disable_times

        self.norm_single = nn.LayerNorm(2 * token_s)
        self.single_embed = nn.Linear(2 * token_s, 2 * token_s)
        if not self.disable_times:
            self.fourier_embed = FourierEmbedding(dim_fourier)
            self.norm_fourier = nn.LayerNorm(dim_fourier)
            self.fourier_to_single = LinearNoBias(dim_fourier, 2 * token_s)

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(
                dim=2 * token_s, hidden=transition_expansion_factor * 2 * token_s
            )
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        times,  # Float[' b'],
        s_trunk,  # Float['b n ts'],
        s_inputs,  # Float['b n ts'],
    ):  # -> Float['b n 2ts']:
        s = torch.cat((s_trunk, s_inputs), dim=-1)
        s = self.single_embed(self.norm_single(s))
        if not self.disable_times:
            fourier_embed = self.fourier_embed(
                times
            )  # note: sigma rescaling done in diffusion module
            normed_fourier = self.norm_fourier(fourier_embed)
            fourier_to_single = self.fourier_to_single(normed_fourier)

            s = rearrange(fourier_to_single, "b d -> b 1 d") + s

        for transition in self.transitions:
            s = transition(s) + s

        return s, normed_fourier if not self.disable_times else None


class PairwiseConditioning(Module):
    """Algorithm 21."""

    def __init__(
        self,
        token_z,
        dim_token_rel_pos_feats,
        num_transitions=2,
        transition_expansion_factor=2,
    ):
        super().__init__()

        self.dim_pairwise_init_proj = nn.Sequential(
            nn.LayerNorm(token_z + dim_token_rel_pos_feats),
            LinearNoBias(token_z + dim_token_rel_pos_feats, token_z),
        )

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(
                dim=token_z, hidden=transition_expansion_factor * token_z
            )
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        z_trunk,  # Float['b n n tz'],
        token_rel_pos_feats,  # Float['b n n 3'],
    ):  # -> Float['b n n tz']:
        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.dim_pairwise_init_proj(z)

        for transition in self.transitions:
            z = transition(z) + z

        return z


def get_indexing_matrix(K, W, H, device):
    assert W % 2 == 0
    assert H % (W // 2) == 0

    h = H // (W // 2)
    assert h % 2 == 0

    arange = torch.arange(2 * K, device=device)
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(
        min=0, max=h + 1
    )
    index = index.view(K, 2, 2 * K)[:, 0, :]
    onehot = one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    return onehot.reshape(2 * K, h * K).float()


def single_to_keys(single, indexing_matrix, W, H):
    B, N, D = single.shape
    K = N // W
    single = single.view(B, 2 * K, W // 2, D)
    return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
        B, K, H, D
    )  # j = 2K, i = W//2, k = h * K


class AtomEncoder(Module):
    def __init__(
        self,
        atom_s,
        atom_z,
        token_s,
        token_z,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_feature_dim,
        structure_prediction=True,
        use_no_atom_char=False,
        use_atom_backbone_feat=False,
        use_residue_feats_atoms=False,
    ):
        super().__init__()

        self.embed_atom_features = Linear(atom_feature_dim, atom_s)
        self.embed_atompair_ref_pos = LinearNoBias(3, atom_z)
        self.embed_atompair_ref_dist = LinearNoBias(1, atom_z)
        self.embed_atompair_mask = LinearNoBias(1, atom_z)
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.use_no_atom_char = use_no_atom_char
        self.use_atom_backbone_feat = use_atom_backbone_feat
        self.use_residue_feats_atoms = use_residue_feats_atoms

        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.s_to_c_trans = nn.Sequential(
                nn.LayerNorm(token_s), LinearNoBias(token_s, atom_s)
            )
            init.final_init_(self.s_to_c_trans[1].weight)

            self.z_to_p_trans = nn.Sequential(
                nn.LayerNorm(token_z), LinearNoBias(token_z, atom_z)
            )
            init.final_init_(self.z_to_p_trans[1].weight)

        self.c_to_p_trans_k = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )
        init.final_init_(self.c_to_p_trans_k[1].weight)

        self.c_to_p_trans_q = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )
        init.final_init_(self.c_to_p_trans_q[1].weight)

        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
        )
        init.final_init_(self.p_mlp[5].weight)

    def forward(
        self,
        feats,
        s_trunk=None,  # Float['bm n ts'],
        z=None,  # Float['bm n n tz'],
    ):
        with torch.autocast("cuda", enabled=False):
            B, N, _ = feats["ref_pos"].shape
            atom_mask = feats["atom_pad_mask"].bool()  # Bool['b m'],

            atom_ref_pos = feats["ref_pos"]  # Float['b m 3'],
            atom_uid = feats["ref_space_uid"]  # Long['b m'],

            atom_feats = [
                atom_ref_pos,
                feats["ref_charge"].unsqueeze(-1),
                feats["ref_element"],
            ]
            if not self.use_no_atom_char:
                atom_feats.append(feats["ref_atom_name_chars"].reshape(B, N, 4 * 64))
            if self.use_atom_backbone_feat:
                atom_feats.append(feats["atom_backbone_feat"])
            if self.use_residue_feats_atoms:
                res_feats = torch.cat(
                    [
                        feats["res_type"],
                        feats["modified"].unsqueeze(-1),
                        one_hot(feats["mol_type"], num_classes=4).float(),
                    ],
                    dim=-1,
                )
                atom_to_token = feats["atom_to_token"].float()
                atom_res_feats = torch.bmm(atom_to_token, res_feats)
                atom_feats.append(atom_res_feats)

            atom_feats = torch.cat(atom_feats, dim=-1)

            c = self.embed_atom_features(atom_feats)

            # note we are already creating the windows to make it more efficient
            W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
            B, N = c.shape[:2]
            K = N // W
            keys_indexing_matrix = get_indexing_matrix(K, W, H, c.device)
            to_keys = partial(
                single_to_keys, indexing_matrix=keys_indexing_matrix, W=W, H=H
            )

            atom_ref_pos_queries = atom_ref_pos.view(B, K, W, 1, 3)
            atom_ref_pos_keys = to_keys(atom_ref_pos).view(B, K, 1, H, 3)

            d = atom_ref_pos_keys - atom_ref_pos_queries  # Float['b k w h 3']
            d_norm = torch.sum(d * d, dim=-1, keepdim=True)  # Float['b k w h 1']
            d_norm = 1 / (
                1 + d_norm
            )  # AF3 feeds in the reciprocal of the distance norm

            atom_mask_queries = atom_mask.view(B, K, W, 1)
            atom_mask_keys = (
                to_keys(atom_mask.unsqueeze(-1).float()).view(B, K, 1, H).bool()
            )
            atom_uid_queries = atom_uid.view(B, K, W, 1)
            atom_uid_keys = (
                to_keys(atom_uid.unsqueeze(-1).float()).view(B, K, 1, H).long()
            )
            v = (
                (
                    atom_mask_queries
                    & atom_mask_keys
                    & (atom_uid_queries == atom_uid_keys)
                )
                .float()
                .unsqueeze(-1)
            )  # Bool['b k w h 1']

            p = self.embed_atompair_ref_pos(d) * v
            p = p + self.embed_atompair_ref_dist(d_norm) * v
            p = p + self.embed_atompair_mask(v) * v

            q = c

            if self.structure_prediction:
                # run only in structure model not in initial encoding
                atom_to_token = feats["atom_to_token"].float()  # Long['b m n'],

                s_to_c = self.s_to_c_trans(s_trunk.float())
                s_to_c = torch.bmm(atom_to_token, s_to_c)
                c = c + s_to_c.to(c)

                atom_to_token_queries = atom_to_token.view(
                    B, K, W, atom_to_token.shape[-1]
                )
                atom_to_token_keys = to_keys(atom_to_token)
                z_to_p = self.z_to_p_trans(z.float())
                z_to_p = torch.einsum(
                    "bijd,bwki,bwlj->bwkld",
                    z_to_p,
                    atom_to_token_queries,
                    atom_to_token_keys,
                )
                p = p + z_to_p.to(p)

            p = p + self.c_to_p_trans_q(c.view(B, K, W, 1, c.shape[-1]))
            p = p + self.c_to_p_trans_k(to_keys(c).view(B, K, 1, H, c.shape[-1]))
            p = p + self.p_mlp(p)
        return q, c, p, to_keys


class AtomAttentionEncoder(Module):
    def __init__(
        self,
        atom_s,
        token_s,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        structure_prediction=True,
        activation_checkpointing=False,
        transformer_post_layer_norm=False,
    ):
        super().__init__()

        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.r_to_q_trans = LinearNoBias(3, atom_s)
            init.final_init_(self.r_to_q_trans.weight)

        self.atom_encoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            activation_checkpointing=activation_checkpointing,
            post_layer_norm=transformer_post_layer_norm,
        )

        self.atom_to_token_trans = nn.Sequential(
            LinearNoBias(atom_s, 2 * token_s if structure_prediction else token_s),
            nn.ReLU(),
        )

    def forward(
        self,
        feats,
        q,
        c,
        atom_enc_bias,
        to_keys,
        r=None,  # Float['bm m 3'],
        multiplicity=1,
    ):
        B, N, _ = feats["ref_pos"].shape
        atom_mask = feats["atom_pad_mask"].bool()  # Bool['b m'],

        if self.structure_prediction:
            # only here the multiplicity kicks in because we use the different positions r
            q = q.repeat_interleave(multiplicity, 0)
            r_to_q = self.r_to_q_trans(r)
            q = q + r_to_q

        c = c.repeat_interleave(multiplicity, 0)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        q = self.atom_encoder(
            q=q,
            mask=atom_mask,
            c=c,
            bias=atom_enc_bias,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        with torch.autocast("cuda", enabled=False):
            q_to_a = self.atom_to_token_trans(q).float()
            atom_to_token = feats["atom_to_token"].float()
            atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
            atom_to_token_mean = atom_to_token / (
                atom_to_token.sum(dim=1, keepdim=True) + 1e-6
            )
            a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        a = a.to(q)

        return a, q, c, to_keys


class AtomAttentionDecoder(Module):
    """Algorithm 6."""

    def __init__(
        self,
        atom_s,
        token_s,
        attn_window_queries,
        attn_window_keys,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        activation_checkpointing=False,
        transformer_post_layer_norm=False,
    ):
        super().__init__()

        self.a_to_q_trans = LinearNoBias(2 * token_s, atom_s)
        init.final_init_(self.a_to_q_trans.weight)

        self.atom_decoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            depth=atom_decoder_depth,
            heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            post_layer_norm=transformer_post_layer_norm,
        )

        if transformer_post_layer_norm:
            self.atom_feat_to_atom_pos_update = LinearNoBias(atom_s, 3)
            init.final_init_(self.atom_feat_to_atom_pos_update.weight)
        else:
            self.atom_feat_to_atom_pos_update = nn.Sequential(
                nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3)
            )
            init.final_init_(self.atom_feat_to_atom_pos_update[1].weight)

    def forward(
        self,
        a,  # Float['bm n 2ts'],
        q,  # Float['bm m as'],
        c,  # Float['bm m as'],
        atom_dec_bias,  # Float['bm m m az'],
        feats,
        to_keys,
        multiplicity=1,
    ):
        with torch.autocast("cuda", enabled=False):
            atom_to_token = feats["atom_to_token"].float()
            atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

            a_to_q = self.a_to_q_trans(a.float())
            a_to_q = torch.bmm(atom_to_token, a_to_q)

        q = q + a_to_q.to(q)
        atom_mask = feats["atom_pad_mask"]  # Bool['b m'],
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        q = self.atom_decoder(
            q=q,
            mask=atom_mask,
            c=c,
            bias=atom_dec_bias,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        r_update = self.atom_feat_to_atom_pos_update(q)
        return r_update

# ---- transformersv2.py ----

# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang


class AdaLN(Module):
    """Algorithm 26"""

    def __init__(self, dim, dim_single_cond):
        super().__init__()
        self.a_norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.s_norm = nn.LayerNorm(dim_single_cond, bias=False)
        self.s_scale = nn.Linear(dim_single_cond, dim)
        self.s_bias = LinearNoBias(dim_single_cond, dim)

    def forward(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        a = F.sigmoid(self.s_scale(s)) * a + self.s_bias(s)
        return a


class ConditionedTransitionBlock(Module):
    """Algorithm 25"""

    def __init__(self, dim_single, dim_single_cond, expansion_factor=2):
        super().__init__()

        self.adaln = AdaLN(dim_single, dim_single_cond)

        dim_inner = int(dim_single * expansion_factor)
        self.swish_gate = Sequential(
            LinearNoBias(dim_single, dim_inner * 2),
            SwiGLU(),
        )
        self.a_to_b = LinearNoBias(dim_single, dim_inner)
        self.b_to_a = LinearNoBias(dim_inner, dim_single)

        output_projection_linear = Linear(dim_single_cond, dim_single)
        nn.init.zeros_(output_projection_linear.weight)
        nn.init.constant_(output_projection_linear.bias, -2.0)

        self.output_projection = nn.Sequential(output_projection_linear, nn.Sigmoid())

    def forward(
        self,
        a,  # Float['... d']
        s,
    ):  # -> Float['... d']:
        a = self.adaln(a, s)
        b = self.swish_gate(a) * self.a_to_b(a)
        a = self.output_projection(s) * self.b_to_a(b)

        return a


class DiffusionTransformer(Module):
    """Algorithm 23"""

    def __init__(
        self,
        depth,
        heads,
        dim=384,
        dim_single_cond=None,
        pair_bias_attn=True,
        activation_checkpointing=False,
        post_layer_norm=False,
    ):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        dim_single_cond = default(dim_single_cond, dim)
        self.pair_bias_attn = pair_bias_attn

        self.layers = ModuleList()
        for _ in range(depth):
            self.layers.append(
                DiffusionTransformerLayer(
                    heads,
                    dim,
                    dim_single_cond,
                    post_layer_norm,
                )
            )

    def forward(
        self,
        a,  # Float['bm n d'],
        s,  # Float['bm n ds'],
        bias=None,  # Float['b n n dp']
        mask=None,  # Bool['b n'] | None = None
        to_keys=None,
        multiplicity=1,
    ):
        if self.pair_bias_attn:
            B, N, M, D = bias.shape
            L = len(self.layers)
            bias = bias.view(B, N, M, L, D // L)

        for i, layer in enumerate(self.layers):
            if self.pair_bias_attn:
                bias_l = bias[:, :, :, i]
            else:
                bias_l = None

            if self.activation_checkpointing and self.training:
                a = torch.utils.checkpoint.checkpoint(
                    layer,
                    a,
                    s,
                    bias_l,
                    mask,
                    to_keys,
                    multiplicity,
                )

            else:
                a = layer(
                    a,  # Float['bm n d'],
                    s,  # Float['bm n ds'],
                    bias_l,  # Float['b n n dp']
                    mask,  # Bool['b n'] | None = None
                    to_keys,
                    multiplicity,
                )
        return a


class DiffusionTransformerLayer(Module):
    """Algorithm 23"""

    def __init__(
        self,
        heads,
        dim=384,
        dim_single_cond=None,
        post_layer_norm=False,
    ):
        super().__init__()

        dim_single_cond = default(dim_single_cond, dim)

        self.adaln = AdaLN(dim, dim_single_cond)
        self.pair_bias_attn = AttentionPairBias(
            c_s=dim, num_heads=heads, compute_pair_bias=False
        )

        self.output_projection_linear = Linear(dim_single_cond, dim)
        nn.init.zeros_(self.output_projection_linear.weight)
        nn.init.constant_(self.output_projection_linear.bias, -2.0)

        self.output_projection = nn.Sequential(
            self.output_projection_linear, nn.Sigmoid()
        )
        self.transition = ConditionedTransitionBlock(
            dim_single=dim, dim_single_cond=dim_single_cond
        )

        if post_layer_norm:
            self.post_lnorm = nn.LayerNorm(dim)
        else:
            self.post_lnorm = nn.Identity()

    def forward(
        self,
        a,  # Float['bm n d'],
        s,  # Float['bm n ds'],
        bias=None,  # Float['b n n dp']
        mask=None,  # Bool['b n'] | None = None
        to_keys=None,
        multiplicity=1,
    ):
        b = self.adaln(a, s)

        k_in = b
        if to_keys is not None:
            k_in = to_keys(b)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)

        if self.pair_bias_attn:
            b = self.pair_bias_attn(
                s=b,
                z=bias,
                mask=mask,
                multiplicity=multiplicity,
                k_in=k_in,
            )
        else:
            b = self.no_pair_bias_attn(s=b, mask=mask, k_in=k_in)

        b = self.output_projection(s) * b

        a = a + b
        a = a + self.transition(a, s)

        a = self.post_lnorm(a)
        return a


class AtomTransformer(Module):
    """Algorithm 7"""

    def __init__(
        self,
        attn_window_queries,
        attn_window_keys,
        **diffusion_transformer_kwargs,
    ):
        super().__init__()
        self.attn_window_queries = attn_window_queries
        self.attn_window_keys = attn_window_keys
        self.diffusion_transformer = DiffusionTransformer(
            **diffusion_transformer_kwargs
        )

    def forward(
        self,
        q,  # Float['b m d'],
        c,  # Float['b m ds'],
        bias,  # Float['b m m dp']
        to_keys,
        mask,  # Bool['b m'] | None = None
        multiplicity=1,
    ):
        W = self.attn_window_queries
        H = self.attn_window_keys

        B, N, D = q.shape
        NW = N // W

        # reshape tokens
        q = q.view((B * NW, W, -1))
        c = c.view((B * NW, W, -1))
        mask = mask.view(B * NW, W)
        bias = bias.repeat_interleave(multiplicity, 0)
        bias = bias.view((bias.shape[0] * NW, W, H, -1))

        to_keys_new = lambda x: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)

        # main transformer
        q = self.diffusion_transformer(
            a=q,
            s=c,
            bias=bias,
            mask=mask.float(),
            multiplicity=1, # bias term already expanded with multiplicity
            to_keys=to_keys_new,
        )

        q = q.view((B, NW * W, D))
        return q

# ---- trunkv2.py ----


class ContactConditioning(nn.Module):
    def __init__(self, token_z: int, cutoff_min: float, cutoff_max: float):
        super().__init__()

        self.fourier_embedding = FourierEmbedding(token_z)
        self.encoder = nn.Linear(
            token_z + len(const.contact_conditioning_info) - 1, token_z
        )
        self.encoding_unspecified = nn.Parameter(torch.zeros(token_z))
        self.encoding_unselected = nn.Parameter(torch.zeros(token_z))
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max

    def forward(self, feats):
        assert const.contact_conditioning_info["UNSPECIFIED"] == 0
        assert const.contact_conditioning_info["UNSELECTED"] == 1
        contact_conditioning = feats["contact_conditioning"][:, :, :, 2:]
        contact_threshold = feats["contact_threshold"]
        contact_threshold_normalized = (contact_threshold - self.cutoff_min) / (
            self.cutoff_max - self.cutoff_min
        )
        contact_threshold_fourier = self.fourier_embedding(
            contact_threshold_normalized.flatten()
        ).reshape(contact_threshold_normalized.shape + (-1,))

        contact_conditioning = torch.cat(
            [
                contact_conditioning,
                contact_threshold_normalized.unsqueeze(-1),
                contact_threshold_fourier,
            ],
            dim=-1,
        )
        contact_conditioning = self.encoder(contact_conditioning)

        contact_conditioning = (
            contact_conditioning
            * (
                1
                - feats["contact_conditioning"][:, :, :, 0:2].sum(dim=-1, keepdim=True)
            )
            + self.encoding_unspecified * feats["contact_conditioning"][:, :, :, 0:1]
            + self.encoding_unselected * feats["contact_conditioning"][:, :, :, 1:2]
        )
        return contact_conditioning


class InputEmbedder(nn.Module):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        atom_encoder_depth: int,
        atom_encoder_heads: int,
        activation_checkpointing: bool = False,
        add_method_conditioning: bool = False,
        add_modified_flag: bool = False,
        add_cyclic_flag: bool = False,
        add_mol_type_feat: bool = False,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
    ) -> None:
        """Initialize the input embedder.

        Parameters
        ----------
        atom_s : int
            The atom embedding size.
        atom_z : int
            The atom pairwise embedding size.
        token_s : int
            The token embedding size.

        """
        super().__init__()
        self.token_s = token_s
        self.add_method_conditioning = add_method_conditioning
        self.add_modified_flag = add_modified_flag
        self.add_cyclic_flag = add_cyclic_flag
        self.add_mol_type_feat = add_mol_type_feat

        self.atom_encoder = AtomEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=False,
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )

        self.atom_enc_proj_z = nn.Sequential(
            nn.LayerNorm(atom_z),
            nn.Linear(atom_z, atom_encoder_depth * atom_encoder_heads, bias=False),
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=False,
            activation_checkpointing=activation_checkpointing,
        )

        self.res_type_encoding = nn.Linear(const.num_tokens, token_s, bias=False)
        self.msa_profile_encoding = nn.Linear(const.num_tokens + 1, token_s, bias=False)

        if add_method_conditioning:
            self.method_conditioning_init = nn.Embedding(
                const.num_method_types, token_s
            )
            self.method_conditioning_init.weight.data.fill_(0)
        if add_modified_flag:
            self.modified_conditioning_init = nn.Embedding(2, token_s)
            self.modified_conditioning_init.weight.data.fill_(0)
        if add_cyclic_flag:
            self.cyclic_conditioning_init = nn.Linear(1, token_s, bias=False)
            self.cyclic_conditioning_init.weight.data.fill_(0)
        if add_mol_type_feat:
            self.mol_type_conditioning_init = nn.Embedding(
                len(const.chain_type_ids), token_s
            )
            self.mol_type_conditioning_init.weight.data.fill_(0)

    def forward(self, feats: dict[str, Tensor], affinity: bool = False) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        feats : dict[str, Tensor]
            Input features

        Returns
        -------
        Tensor
            The embedded tokens.

        """
        # Load relevant features
        res_type = feats["res_type"].float()
        if affinity:
            profile = feats["profile_affinity"]
            deletion_mean = feats["deletion_mean_affinity"].unsqueeze(-1)
        else:
            profile = feats["profile"]
            deletion_mean = feats["deletion_mean"].unsqueeze(-1)

        # Compute input embedding
        q, c, p, to_keys = self.atom_encoder(feats)
        atom_enc_bias = self.atom_enc_proj_z(p)
        a, _, _, _ = self.atom_attention_encoder(
            feats=feats,
            q=q,
            c=c,
            atom_enc_bias=atom_enc_bias,
            to_keys=to_keys,
        )

        s = (
            a
            + self.res_type_encoding(res_type)
            + self.msa_profile_encoding(torch.cat([profile, deletion_mean], dim=-1))
        )

        if self.add_method_conditioning:
            s = s + self.method_conditioning_init(feats["method_feature"])
        if self.add_modified_flag:
            s = s + self.modified_conditioning_init(feats["modified"])
        if self.add_cyclic_flag:
            cyclic = feats["cyclic_period"].clamp(max=1.0).unsqueeze(-1)
            s = s + self.cyclic_conditioning_init(cyclic)
        if self.add_mol_type_feat:
            s = s + self.mol_type_conditioning_init(feats["mol_type"])

        return s


class TemplateModule(nn.Module):
    """Template module."""

    def __init__(
        self,
        token_z: int,
        template_dim: int,
        template_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        num_bins: int = 38,
        **kwargs,
    ) -> None:
        """Initialize the template module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.

        """
        super().__init__()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.relu = nn.ReLU()
        self.z_norm = nn.LayerNorm(token_z)
        self.v_norm = nn.LayerNorm(template_dim)
        self.z_proj = nn.Linear(token_z, template_dim, bias=False)
        self.a_proj = nn.Linear(
            const.num_tokens * 2 + num_bins + 5,
            template_dim,
            bias=False,
        )
        self.u_proj = nn.Linear(template_dim, token_z, bias=False)
        _, _, PairformerNoSeqModule_, _ = _get_pytorch_modules()
        self.pairformer = PairformerNoSeqModule_(
            template_dim,
            num_blocks=template_blocks,
            dropout=dropout,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            post_layer_norm=post_layer_norm,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(
        self,
        z: Tensor,
        feats: dict[str, Tensor],
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings
        feats : dict[str, Tensor]
            Input features
        pair_mask : Tensor
            The pair mask

        Returns
        -------
        Tensor
            The updated pairwise embeddings.

        """
        # Load relevant features
        asym_id = feats["asym_id"]
        res_type = feats["template_restype"]
        frame_rot = feats["template_frame_rot"]
        frame_t = feats["template_frame_t"]
        frame_mask = feats["template_mask_frame"]
        cb_coords = feats["template_cb"]
        ca_coords = feats["template_ca"]
        cb_mask = feats["template_mask_cb"]
        template_mask = feats["template_mask"].any(dim=2).float()
        num_templates = template_mask.sum(dim=1)
        num_templates = num_templates.clamp(min=1)

        # Compute pairwise masks
        b_cb_mask = cb_mask[:, :, :, None] * cb_mask[:, :, None, :]
        b_frame_mask = frame_mask[:, :, :, None] * frame_mask[:, :, None, :]

        b_cb_mask = b_cb_mask[..., None]
        b_frame_mask = b_frame_mask[..., None]

        # Compute asym mask, template features only attend within the same chain
        B, T = res_type.shape[:2]  # noqa: N806
        asym_mask = (asym_id[:, :, None] == asym_id[:, None, :]).float()
        asym_mask = asym_mask[:, None].expand(-1, T, -1, -1)

        # Compute template features
        with torch.autocast(device_type="cuda", enabled=False):
            # Compute distogram
            cb_dists = torch.cdist(cb_coords, cb_coords)
            boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1)
            boundaries = boundaries.to(cb_dists.device)
            distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
            distogram = one_hot(distogram, num_classes=self.num_bins)

            # Compute unit vector in each frame
            frame_rot = frame_rot.unsqueeze(2).transpose(-1, -2)
            frame_t = frame_t.unsqueeze(2).unsqueeze(-1)
            ca_coords = ca_coords.unsqueeze(3).unsqueeze(-1)
            vector = torch.matmul(frame_rot, (ca_coords - frame_t))
            norm = torch.norm(vector, dim=-1, keepdim=True)
            unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector))
            unit_vector = unit_vector.squeeze(-1)

            # Concatenate input features
            a_tij = [distogram, b_cb_mask, unit_vector, b_frame_mask]
            a_tij = torch.cat(a_tij, dim=-1)
            a_tij = a_tij * asym_mask.unsqueeze(-1)

            res_type_i = res_type[:, :, :, None]
            res_type_j = res_type[:, :, None, :]
            res_type_i = res_type_i.expand(-1, -1, -1, res_type.size(2), -1)
            res_type_j = res_type_j.expand(-1, -1, res_type.size(2), -1, -1)
            a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)
            a_tij = self.a_proj(a_tij)

        # Expand mask
        pair_mask = pair_mask[:, None].expand(-1, T, -1, -1)
        pair_mask = pair_mask.reshape(B * T, *pair_mask.shape[2:])

        # Compute input projections
        v = self.z_proj(self.z_norm(z[:, None])) + a_tij
        v = v.view(B * T, *v.shape[2:])
        v = v + self.pairformer(v, pair_mask, use_kernels=use_kernels)
        v = self.v_norm(v)
        v = v.view(B, T, *v.shape[1:])

        # Aggregate templates
        template_mask = template_mask[:, :, None, None, None]
        num_templates = num_templates[:, None, None, None]
        u = (v * template_mask).sum(dim=1) / num_templates.to(v)

        # Compute output projection
        u = self.u_proj(self.relu(u))
        return u


class BFactorModule(nn.Module):
    """BFactor Module."""

    def __init__(self, token_s: int, num_bins: int) -> None:
        """Initialize the bfactor module.

        Parameters
        ----------
        token_s : int
            The token embedding size.

        """
        super().__init__()
        self.bfactor = nn.Linear(token_s, num_bins)
        self.num_bins = num_bins

    def forward(self, s: Tensor) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        s : Tensor
            The sequence embeddings

        Returns
        -------
        Tensor
            The predicted bfactor histogram.

        """
        return self.bfactor(s)


class DistogramModule(nn.Module):
    """Distogram Module."""

    def __init__(self, token_z: int, num_bins: int, num_distograms: int = 1) -> None:
        """Initialize the distogram module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.

        """
        super().__init__()
        self.distogram = nn.Linear(token_z, num_distograms * num_bins)
        self.num_distograms = num_distograms
        self.num_bins = num_bins

    def forward(self, z: Tensor) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings

        Returns
        -------
        Tensor
            The predicted distogram.

        """
        z = z + z.transpose(1, 2)
        return self.distogram(z).reshape(
            z.shape[0], z.shape[1], z.shape[2], self.num_distograms, self.num_bins
        )

# ---- diffusion_conditioning.py ----


class DiffusionConditioning(Module):
    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
    ) -> None:
        super().__init__()

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_encoder = AtomEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=True,
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )

        self.atom_enc_proj_z = nn.ModuleList()
        for _ in range(atom_encoder_depth):
            self.atom_enc_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(atom_z),
                    nn.Linear(atom_z, atom_encoder_heads, bias=False),
                )
            )

        self.atom_dec_proj_z = nn.ModuleList()
        for _ in range(atom_decoder_depth):
            self.atom_dec_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(atom_z),
                    nn.Linear(atom_z, atom_decoder_heads, bias=False),
                )
            )

        self.token_trans_proj_z = nn.ModuleList()
        for _ in range(token_transformer_depth):
            self.token_trans_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(token_z),
                    nn.Linear(token_z, token_transformer_heads, bias=False),
                )
            )

    def forward(
        self,
        s_trunk,  # Float['b n ts']
        z_trunk,  # Float['b n n tz']
        relative_position_encoding,  # Float['b n n tz']
        feats,
    ):
        z = self.pairwise_conditioner(
            z_trunk,
            relative_position_encoding,
        )

        q, c, p, to_keys = self.atom_encoder(
            feats=feats,
            s_trunk=s_trunk,  # Float['b n ts'],
            z=z,  # Float['b n n tz'],
        )

        atom_enc_bias = []
        for layer in self.atom_enc_proj_z:
            atom_enc_bias.append(layer(p))
        atom_enc_bias = torch.cat(atom_enc_bias, dim=-1)

        atom_dec_bias = []
        for layer in self.atom_dec_proj_z:
            atom_dec_bias.append(layer(p))
        atom_dec_bias = torch.cat(atom_dec_bias, dim=-1)

        token_trans_bias = []
        for layer in self.token_trans_proj_z:
            token_trans_bias.append(layer(z))
        token_trans_bias = torch.cat(token_trans_bias, dim=-1)

        return q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias

# ---- diffusionv2.py ----

# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang


class ConfidenceHeads(nn.Module):
    def __init__(
        self,
        token_s,
        token_z,
        num_plddt_bins=50,
        num_pde_bins=64,
        num_pae_bins=64,
        token_level_confidence=True,
        use_separate_heads: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.max_num_atoms_per_token = 23
        self.token_level_confidence = token_level_confidence
        self.use_separate_heads = use_separate_heads

        if self.use_separate_heads:
            self.to_pae_intra_logits = LinearNoBias(token_z, num_pae_bins)
            self.to_pae_inter_logits = LinearNoBias(token_z, num_pae_bins)
        else:
            self.to_pae_logits = LinearNoBias(token_z, num_pae_bins)

        if self.use_separate_heads:
            self.to_pde_intra_logits = LinearNoBias(token_z, num_pde_bins)
            self.to_pde_inter_logits = LinearNoBias(token_z, num_pde_bins)
        else:
            self.to_pde_logits = LinearNoBias(token_z, num_pde_bins)

        if self.token_level_confidence:
            self.to_plddt_logits = LinearNoBias(token_s, num_plddt_bins)
            self.to_resolved_logits = LinearNoBias(token_s, 2)
        else:
            self.to_plddt_logits = LinearNoBias(
                token_s, num_plddt_bins * self.max_num_atoms_per_token
            )
            self.to_resolved_logits = LinearNoBias(
                token_s, 2 * self.max_num_atoms_per_token
            )

    def forward(
        self,
        s,  # Float['b n ts']
        z,  # Float['b n n tz']
        x_pred,  # Float['bm m 3']
        d,
        feats,
        pred_distogram_logits,
        multiplicity=1,
    ):
        if self.use_separate_heads:
            asym_id_token = feats["asym_id"]
            is_same_chain = asym_id_token.unsqueeze(-1) == asym_id_token.unsqueeze(-2)
            is_different_chain = ~is_same_chain

        if self.use_separate_heads:
            pae_intra_logits = self.to_pae_intra_logits(z)
            pae_intra_logits = pae_intra_logits * is_same_chain.float().unsqueeze(-1)

            pae_inter_logits = self.to_pae_inter_logits(z)
            pae_inter_logits = pae_inter_logits * is_different_chain.float().unsqueeze(
                -1
            )

            pae_logits = pae_inter_logits + pae_intra_logits
        else:
            pae_logits = self.to_pae_logits(z)

        if self.use_separate_heads:
            pde_intra_logits = self.to_pde_intra_logits(z + z.transpose(1, 2))
            pde_intra_logits = pde_intra_logits * is_same_chain.float().unsqueeze(-1)

            pde_inter_logits = self.to_pde_inter_logits(z + z.transpose(1, 2))
            pde_inter_logits = pde_inter_logits * is_different_chain.float().unsqueeze(
                -1
            )

            pde_logits = pde_inter_logits + pde_intra_logits
        else:
            pde_logits = self.to_pde_logits(z + z.transpose(1, 2))
        resolved_logits = self.to_resolved_logits(s)
        plddt_logits = self.to_plddt_logits(s)

        ligand_weight = 20
        non_interface_weight = 1
        interface_weight = 10

        token_type = feats["mol_type"]
        token_type = token_type.repeat_interleave(multiplicity, 0)
        is_ligand_token = (token_type == const.chain_type_ids["NONPOLYMER"]).float()

        if self.token_level_confidence:
            plddt = compute_aggregated_metric(plddt_logits)
            token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
            complex_plddt = (plddt * token_pad_mask).sum(dim=-1) / token_pad_mask.sum(
                dim=-1
            )

            is_contact = (d < 8).float()
            is_different_chain = (
                feats["asym_id"].unsqueeze(-1) != feats["asym_id"].unsqueeze(-2)
            ).float()
            is_different_chain = is_different_chain.repeat_interleave(multiplicity, 0)
            token_interface_mask = torch.max(
                is_contact * is_different_chain * (1 - is_ligand_token).unsqueeze(-1),
                dim=-1,
            ).values
            token_non_interface_mask = (1 - token_interface_mask) * (
                1 - is_ligand_token
            )
            iplddt_weight = (
                is_ligand_token * ligand_weight
                + token_interface_mask * interface_weight
                + token_non_interface_mask * non_interface_weight
            )
            complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(
                dim=-1
            ) / torch.sum(token_pad_mask * iplddt_weight, dim=-1)

        else:
            # token to atom conversion for resolved logits
            B, N, _ = resolved_logits.shape
            resolved_logits = resolved_logits.reshape(
                B, N, self.max_num_atoms_per_token, 2
            )

            arange_max_num_atoms = (
                torch.arange(self.max_num_atoms_per_token)
                .reshape(1, 1, -1)
                .to(resolved_logits.device)
            )
            max_num_atoms_mask = (
                feats["atom_to_token"].sum(1).unsqueeze(-1) > arange_max_num_atoms
            )
            resolved_logits = resolved_logits[:, max_num_atoms_mask.squeeze(0)]
            resolved_logits = pad(
                resolved_logits,
                (
                    0,
                    0,
                    0,
                    int(
                        feats["atom_pad_mask"].shape[1]
                        - feats["atom_pad_mask"].sum().item()
                    ),
                ),
                value=0,
            )
            plddt_logits = plddt_logits.reshape(B, N, self.max_num_atoms_per_token, -1)
            plddt_logits = plddt_logits[:, max_num_atoms_mask.squeeze(0)]
            plddt_logits = pad(
                plddt_logits,
                (
                    0,
                    0,
                    0,
                    int(
                        feats["atom_pad_mask"].shape[1]
                        - feats["atom_pad_mask"].sum().item()
                    ),
                ),
                value=0,
            )
            atom_pad_mask = feats["atom_pad_mask"].repeat_interleave(multiplicity, 0)
            plddt = compute_aggregated_metric(plddt_logits)

            complex_plddt = (plddt * atom_pad_mask).sum(dim=-1) / atom_pad_mask.sum(
                dim=-1
            )
            token_type = feats["mol_type"].float()
            atom_to_token = feats["atom_to_token"].float()
            chain_id_token = feats["asym_id"].float()
            atom_type = torch.bmm(atom_to_token, token_type.unsqueeze(-1)).squeeze(-1)
            is_ligand_atom = (atom_type == const.chain_type_ids["NONPOLYMER"]).float()
            d_atom = torch.cdist(x_pred, x_pred)
            is_contact = (d_atom < 8).float()
            chain_id_atom = torch.bmm(
                atom_to_token, chain_id_token.unsqueeze(-1)
            ).squeeze(-1)
            is_different_chain = (
                chain_id_atom.unsqueeze(-1) != chain_id_atom.unsqueeze(-2)
            ).float()

            atom_interface_mask = torch.max(
                is_contact * is_different_chain * (1 - is_ligand_atom).unsqueeze(-1),
                dim=-1,
            ).values
            atom_non_interface_mask = (1 - atom_interface_mask) * (1 - is_ligand_atom)
            iplddt_weight = (
                is_ligand_atom * ligand_weight
                + atom_interface_mask * interface_weight
                + atom_non_interface_mask * non_interface_weight
            )

            complex_iplddt = (plddt * feats["atom_pad_mask"] * iplddt_weight).sum(
                dim=-1
            ) / torch.sum(feats["atom_pad_mask"] * iplddt_weight, dim=-1)

        # Compute the gPDE and giPDE
        pde = compute_aggregated_metric(pde_logits, end=32)
        pred_distogram_prob = nn.functional.softmax(
            pred_distogram_logits, dim=-1
        ).repeat_interleave(multiplicity, 0)
        contacts = torch.zeros((1, 1, 1, 64), dtype=pred_distogram_prob.dtype).to(
            pred_distogram_prob.device
        )
        contacts[:, :, :, :20] = 1.0
        prob_contact = (pred_distogram_prob * contacts).sum(-1)
        token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        token_pad_pair_mask = (
            token_pad_mask.unsqueeze(-1)
            * token_pad_mask.unsqueeze(-2)
            * (
                1
                - torch.eye(
                    token_pad_mask.shape[1], device=token_pad_mask.device
                ).unsqueeze(0)
            )
        )
        token_pair_mask = token_pad_pair_mask * prob_contact
        complex_pde = (pde * token_pair_mask).sum(dim=(1, 2)) / token_pair_mask.sum(
            dim=(1, 2)
        )
        asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
        token_interface_pair_mask = token_pair_mask * (
            asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
        )
        complex_ipde = (pde * token_interface_pair_mask).sum(dim=(1, 2)) / (
            token_interface_pair_mask.sum(dim=(1, 2)) + 1e-5
        )
        out_dict = dict(
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            pde=pde,
            plddt=plddt,
            complex_plddt=complex_plddt,
            complex_iplddt=complex_iplddt,
            complex_pde=complex_pde,
            complex_ipde=complex_ipde,
        )
        out_dict["pae_logits"] = pae_logits
        out_dict["pae"] = compute_aggregated_metric(pae_logits, end=32)

        try:
            ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm = compute_ptms(
                pae_logits, x_pred, feats, multiplicity
            )
            out_dict["ptm"] = ptm
            out_dict["iptm"] = iptm
            out_dict["ligand_iptm"] = ligand_iptm
            out_dict["protein_iptm"] = protein_iptm
            out_dict["pair_chains_iptm"] = pair_chains_iptm
        except Exception as e:
            print(f"Error in compute_ptms: {e}")
            out_dict["ptm"] = torch.zeros_like(complex_plddt)
            out_dict["iptm"] = torch.zeros_like(complex_plddt)
            out_dict["ligand_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["protein_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["pair_chains_iptm"] = torch.zeros_like(complex_plddt)

        return out_dict

# ---- affinity.py ----


class AffinityHeadsTransformer(nn.Module):
    def __init__(
        self,
        token_z,
        input_token_s,
        num_blocks,
        num_heads,
        activation_checkpointing,
        use_cross_transformer,
        groups={},
    ):
        super().__init__()
        self.affinity_out_mlp = nn.Sequential(
            nn.Linear(token_z, token_z),
            nn.ReLU(),
            nn.Linear(token_z, input_token_s),
            nn.ReLU(),
        )

        self.to_affinity_pred_value = nn.Sequential(
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, 1),
        )

        self.to_affinity_pred_score = nn.Sequential(
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, 1),
        )
        self.to_affinity_logits_binary = nn.Linear(1, 1)

    def forward(
        self,
        z,
        feats,
        multiplicity=1,
    ):
        pad_token_mask = (
            feats["token_pad_mask"].repeat_interleave(multiplicity, 0).unsqueeze(-1)
        )
        rec_mask = (
            (feats["mol_type"] == 0).repeat_interleave(multiplicity, 0).unsqueeze(-1)
        )
        rec_mask = rec_mask * pad_token_mask
        lig_mask = (
            feats["affinity_token_mask"]
            .repeat_interleave(multiplicity, 0)
            .to(torch.bool)
            .unsqueeze(-1)
        ) * pad_token_mask
        cross_pair_mask = (
            lig_mask[:, :, None] * rec_mask[:, None, :]
            + rec_mask[:, :, None] * lig_mask[:, None, :]
            + (lig_mask[:, :, None] * lig_mask[:, None, :])
        ) * (
            1
            - torch.eye(lig_mask.shape[1], device=lig_mask.device)
            .unsqueeze(-1)
            .unsqueeze(0)
        )

        g = torch.sum(z * cross_pair_mask, dim=(1, 2)) / (
            torch.sum(cross_pair_mask, dim=(1, 2)) + 1e-7
        )

        g = self.affinity_out_mlp(g)

        affinity_pred_value = self.to_affinity_pred_value(g).reshape(-1, 1)
        affinity_pred_score = self.to_affinity_pred_score(g).reshape(-1, 1)
        affinity_logits_binary = self.to_affinity_logits_binary(
            affinity_pred_score
        ).reshape(-1, 1)
        out_dict = {
            "affinity_pred_value": affinity_pred_value,
            "affinity_logits_binary": affinity_logits_binary,
        }
        return out_dict


# ---- diffusionv2 loss helpers ----


def smooth_lddt_loss(
    pred_coords,
    true_coords,
    is_nucleotide,
    coords_mask,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    multiplicity: int = 1,
):
    lddt = []
    for i in range(true_coords.shape[0]):
        true_dists = torch.cdist(true_coords[i], true_coords[i])
        is_nucleotide_i = is_nucleotide[i // multiplicity]
        coords_mask_i = coords_mask[i // multiplicity]
        is_nucleotide_pair = is_nucleotide_i.unsqueeze(-1).expand(
            -1, is_nucleotide_i.shape[-1]
        )
        mask = is_nucleotide_pair * (true_dists < nucleic_acid_cutoff).float()
        mask += (1 - is_nucleotide_pair) * (true_dists < other_cutoff).float()
        mask *= 1 - torch.eye(pred_coords.shape[1], device=pred_coords.device)
        mask *= coords_mask_i.unsqueeze(-1)
        mask *= coords_mask_i.unsqueeze(-2)

        valid_pairs = mask.nonzero()
        true_dists_i = true_dists[valid_pairs[:, 0], valid_pairs[:, 1]]
        pred_coords_i1 = pred_coords[i, valid_pairs[:, 0]]
        pred_coords_i2 = pred_coords[i, valid_pairs[:, 1]]
        pred_dists_i = F.pairwise_distance(pred_coords_i1, pred_coords_i2)
        dist_diff_i = torch.abs(true_dists_i - pred_dists_i)
        eps_i = (
            F.sigmoid(0.5 - dist_diff_i)
            + F.sigmoid(1.0 - dist_diff_i)
            + F.sigmoid(2.0 - dist_diff_i)
            + F.sigmoid(4.0 - dist_diff_i)
        ) / 4.0
        lddt_i = eps_i.sum() / (valid_pairs.shape[0] + 1e-5)
        lddt.append(lddt_i)

    return 1.0 - torch.stack(lddt, dim=0).mean(dim=0)


# LinearNoBias defined after Linear
LinearNoBias = partial(TorchLinear, bias=False)

# ============================================================================
# POTENTIALS & SCHEDULES (for steering/guidance)
# ============================================================================

class ParameterSchedule(ABC):
    def compute(self, t):
        raise NotImplementedError


class ExponentialInterpolation(ParameterSchedule):
    def __init__(self, start, end, alpha):
        self.start = start
        self.end = end
        self.alpha = alpha

    def compute(self, t):
        if self.alpha != 0:
            return self.start + (self.end - self.start) * (
                exp(self.alpha * t) - 1
            ) / (exp(self.alpha) - 1)
        else:
            return self.start + (self.end - self.start) * t


class PiecewiseStepFunction(ParameterSchedule):
    def __init__(self, thresholds, values):
        self.thresholds = thresholds
        self.values = values

    def compute(self, t):
        assert len(self.thresholds) > 0
        assert len(self.values) == len(self.thresholds) + 1

        idx = 0
        while idx < len(self.thresholds) and t > self.thresholds[idx]:
            idx += 1
        return self.values[idx]

# ---- potentials.py ----


class Potential(ABC):
    def __init__(
        self,
        parameters: Optional[
            Dict[str, Union[ParameterSchedule, float, int, bool]]
        ] = None,
    ):
        self.parameters = parameters

    def compute(self, coords, feats, parameters):
        index, args, com_args, ref_args, operator_args = self.compute_args(
            feats, parameters
        )

        if index.shape[1] == 0:
            return torch.zeros(coords.shape[:-2], device=coords.device)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_com_index = com_index[atom_pad_mask]
            unpad_coords = coords[..., atom_pad_mask, :]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
        else:
            com_index, atom_pad_mask = None, None

        if ref_args is not None:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = ref_args
            coords = coords[..., ref_atom_index, :]
        else:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = (
                None,
                None,
                None,
                None,
            )

        if operator_args is not None:
            negation_mask, union_index = operator_args
        else:
            negation_mask, union_index = None, None

        value = self.compute_variable(
            coords,
            index,
            ref_coords=ref_coords,
            ref_mask=ref_mask,
            compute_gradient=False,
        )
        energy = self.compute_function(
            value, *args, negation_mask=negation_mask, compute_derivative=False
        )

        if union_index is not None:
            neg_exp_energy = torch.exp(-1 * parameters["union_lambda"] * energy)
            Z = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(neg_exp_energy),
                neg_exp_energy,
                "sum",
            )
            softmax_energy = neg_exp_energy / Z[..., union_index]
            softmax_energy[Z[..., union_index] == 0] = 0
            return (energy * softmax_energy).sum(dim=-1)

        return energy.sum(dim=tuple(range(1, energy.dim())))

    def compute_gradient(self, coords, feats, parameters):
        index, args, com_args, ref_args, operator_args = self.compute_args(
            feats, parameters
        )
        if index.shape[1] == 0:
            return torch.zeros_like(coords)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_coords = coords[..., atom_pad_mask, :]
            unpad_com_index = com_index[atom_pad_mask]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
            com_counts = torch.bincount(com_index[atom_pad_mask])
        else:
            com_index, atom_pad_mask = None, None

        if ref_args is not None:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = ref_args
            coords = coords[..., ref_atom_index, :]
        else:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = (
                None,
                None,
                None,
                None,
            )

        if operator_args is not None:
            negation_mask, union_index = operator_args
        else:
            negation_mask, union_index = None, None

        value, grad_value = self.compute_variable(
            coords,
            index,
            ref_coords=ref_coords,
            ref_mask=ref_mask,
            compute_gradient=True,
        )
        energy, dEnergy = self.compute_function(
            value, 
            *args, negation_mask=negation_mask, compute_derivative=True
        )
        if union_index is not None:
            neg_exp_energy = torch.exp(-1 * parameters["union_lambda"] * energy)
            Z = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(energy),
                neg_exp_energy,
                "sum",
            )
            softmax_energy = neg_exp_energy / Z[..., union_index]
            softmax_energy[Z[..., union_index] == 0] = 0
            f = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(energy),
                energy * softmax_energy,
                "sum",
            )
            dSoftmax = (
                dEnergy
                * softmax_energy
                * (1 + parameters["union_lambda"] * (energy - f[..., union_index]))
            )
            prod = dSoftmax.tile(grad_value.shape[-3]).unsqueeze(
                -1
            ) * grad_value.flatten(start_dim=-3, end_dim=-2)
            if prod.dim() > 3:
                prod = prod.sum(dim=list(range(1, prod.dim() - 2)))
            grad_atom = torch.zeros_like(coords).scatter_reduce(
                -2,
                index.flatten(start_dim=0, end_dim=1)
                .unsqueeze(-1)
                .expand((*coords.shape[:-2], -1, 3)),
                prod,
                "sum",
            )
        else:
            prod = dEnergy.tile(grad_value.shape[-3]).unsqueeze(
                -1
            ) * grad_value.flatten(start_dim=-3, end_dim=-2)
            if prod.dim() > 3:
                prod = prod.sum(dim=list(range(1, prod.dim() - 2)))
            grad_atom = torch.zeros_like(coords).scatter_reduce(
                -2,
                index.flatten(start_dim=0, end_dim=1)
                .unsqueeze(-1)
                .expand((*coords.shape[:-2], -1, 3)),  # 9 x 516 x 3
                prod,
                "sum",
            )

        if com_index is not None:
            grad_atom = grad_atom[..., com_index, :]
        elif ref_token_index is not None:
            grad_atom = grad_atom[..., ref_token_index, :]

        return grad_atom

    def compute_parameters(self, t):
        if self.parameters is None:
            return None
        parameters = {
            name: parameter
            if not isinstance(parameter, ParameterSchedule)
            else parameter.compute(t)
            for name, parameter in self.parameters.items()
        }
        return parameters

    @abstractmethod
    def compute_function(
        self, value, *args, negation_mask=None, compute_derivative=False
    ):
        raise NotImplementedError

    @abstractmethod
    def compute_variable(self, coords, index, compute_gradient=False):
        raise NotImplementedError

    @abstractmethod
    def compute_args(self, t, feats, **parameters):
        raise NotImplementedError

    def get_reference_coords(self, feats, parameters):
        return None, None


class FlatBottomPotential(Potential):
    def compute_function(
        self,
        value,
        k,
        lower_bounds,
        upper_bounds,
        negation_mask=None,
        compute_derivative=False,
    ):
        if lower_bounds is None:
            lower_bounds = torch.full_like(value, float("-inf"))
        if upper_bounds is None:
            upper_bounds = torch.full_like(value, float("inf"))
        lower_bounds = lower_bounds.expand_as(value).clone()
        upper_bounds = upper_bounds.expand_as(value).clone()

        if negation_mask is not None:
            unbounded_below_mask = torch.isneginf(lower_bounds)
            unbounded_above_mask = torch.isposinf(upper_bounds)
            unbounded_mask = unbounded_below_mask + unbounded_above_mask
            assert torch.all(unbounded_mask + negation_mask)
            lower_bounds[~unbounded_above_mask * ~negation_mask] = upper_bounds[
                ~unbounded_above_mask * ~negation_mask
            ]
            upper_bounds[~unbounded_above_mask * ~negation_mask] = float("inf")
            upper_bounds[~unbounded_below_mask * ~negation_mask] = lower_bounds[
                ~unbounded_below_mask * ~negation_mask
            ]
            lower_bounds[~unbounded_below_mask * ~negation_mask] = float("-inf")

        # Ensure all tensors match value dtype to avoid dtype mismatch
        # This is important when using Tenstorrent where features may be bfloat16
        # but value (from distance computation) is float32
        k = k.to(dtype=value.dtype)
        if lower_bounds is not None:
            lower_bounds = lower_bounds.to(dtype=value.dtype)
        if upper_bounds is not None:
            upper_bounds = upper_bounds.to(dtype=value.dtype)

        neg_overflow_mask = value < lower_bounds
        pos_overflow_mask = value > upper_bounds

        energy = torch.zeros_like(value)
        energy[neg_overflow_mask] = (k * (lower_bounds - value))[neg_overflow_mask]
        energy[pos_overflow_mask] = (k * (value - upper_bounds))[pos_overflow_mask]
        if not compute_derivative:
            return energy

        dEnergy = torch.zeros_like(value)
        dEnergy[neg_overflow_mask] = (
            -1 * k.expand_as(neg_overflow_mask)[neg_overflow_mask]
        )
        dEnergy[pos_overflow_mask] = (
            1 * k.expand_as(pos_overflow_mask)[pos_overflow_mask]
        )

        return energy, dEnergy


class ReferencePotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords, ref_mask, compute_gradient=False
    ):
        aligned_ref_coords = weighted_rigid_align(
            ref_coords.float(),
            coords[:, index].float(),
            ref_mask,
            ref_mask,
        )

        r = coords[:, index] - aligned_ref_coords
        r_norm = torch.linalg.norm(r, dim=-1)

        if not compute_gradient:
            return r_norm

        r_hat = r / r_norm.unsqueeze(-1)
        grad = (r_hat * ref_mask.unsqueeze(-1)).unsqueeze(1)
        return r_norm, grad


class DistancePotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_ij_norm = torch.linalg.norm(r_ij, dim=-1)
        r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)

        if not compute_gradient:
            return r_ij_norm

        grad_i = r_hat_ij
        grad_j = -1 * r_hat_ij
        grad = torch.stack((grad_i, grad_j), dim=1)
        return r_ij_norm, grad


class DihedralPotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])

        n_ijk = torch.cross(r_ij, r_kj, dim=-1)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)

        r_kj_norm = torch.linalg.norm(r_kj, dim=-1)
        n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1)
        n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1)

        sign_phi = torch.sign(
            r_kj.unsqueeze(-2) @ torch.cross(n_ijk, n_jkl, dim=-1).unsqueeze(-1)
        ).squeeze(-1, -2)
        phi = sign_phi * torch.arccos(
            torch.clamp(
                (n_ijk.unsqueeze(-2) @ n_jkl.unsqueeze(-1)).squeeze(-1, -2)
                / (n_ijk_norm * n_jkl_norm),
                -1 + 1e-8,
                1 - 1e-8,
            )
        )

        if not compute_gradient:
            return phi

        a = (
            (r_ij.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)
        b = (
            (r_kl.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)

        grad_i = n_ijk * (r_kj_norm / n_ijk_norm**2).unsqueeze(-1)
        grad_l = -1 * n_jkl * (r_kj_norm / n_jkl_norm**2).unsqueeze(-1)
        grad_j = (a - 1) * grad_i - b * grad_l
        grad_k = (b - 1) * grad_l - a * grad_i
        grad = torch.stack((grad_i, grad_j, grad_k, grad_l), dim=1)
        return phi, grad


class AbsDihedralPotential(DihedralPotential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        if not compute_gradient:
            phi = super().compute_variable(
                coords, index, compute_gradient=compute_gradient
            )
            phi = torch.abs(phi)
            return phi

        phi, grad = super().compute_variable(
            coords, index, compute_gradient=compute_gradient
        )
        grad[(phi < 0)[..., None, :, None].expand_as(grad)] *= -1
        phi = torch.abs(phi)

        return phi, grad


class PoseBustersPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats["rdkit_bounds_index"][0]
        lower_bounds = feats["rdkit_lower_bounds"][0].clone()
        upper_bounds = feats["rdkit_upper_bounds"][0].clone()
        bond_mask = feats["rdkit_bounds_bond_mask"][0]
        angle_mask = feats["rdkit_bounds_angle_mask"][0]

        lower_bounds[bond_mask * ~angle_mask] *= 1.0 - parameters["bond_buffer"]
        upper_bounds[bond_mask * ~angle_mask] *= 1.0 + parameters["bond_buffer"]
        lower_bounds[~bond_mask * angle_mask] *= 1.0 - parameters["angle_buffer"]
        upper_bounds[~bond_mask * angle_mask] *= 1.0 + parameters["angle_buffer"]
        lower_bounds[bond_mask * angle_mask] *= 1.0 - min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        upper_bounds[bond_mask * angle_mask] *= 1.0 + min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        lower_bounds[~bond_mask * ~angle_mask] *= 1.0 - parameters["clash_buffer"]
        upper_bounds[~bond_mask * ~angle_mask] = float("inf")

        vdw_radii = torch.zeros(
            const.num_elements, dtype=torch.float32, device=pair_index.device
        )
        vdw_radii[1:119] = torch.tensor(
            const.vdw_radii, dtype=torch.float32, device=pair_index.device
        )
        atom_vdw_radii = (
            feats["ref_element"].float() @ vdw_radii.unsqueeze(-1)
        ).squeeze(-1)[0]
        bond_cutoffs = 0.35 + atom_vdw_radii[pair_index].mean(dim=0)
        lower_bounds[~bond_mask] = torch.max(lower_bounds[~bond_mask], bond_cutoffs[~bond_mask])
        upper_bounds[bond_mask] = torch.min(upper_bounds[bond_mask], bond_cutoffs[bond_mask])

        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class ConnectionsPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats["connected_atom_index"][0]
        lower_bounds = None
        upper_bounds = torch.full(
            (pair_index.shape[1],), parameters["buffer"], device=pair_index.device
        )
        k = torch.ones_like(upper_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class VDWOverlapPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = (chain_sizes > 1)[atom_chain_id]

        vdw_radii = torch.zeros(
            const.num_elements, dtype=torch.float32, device=atom_chain_id.device
        )
        vdw_radii[1:119] = torch.tensor(
            const.vdw_radii, dtype=torch.float32, device=atom_chain_id.device
        )
        atom_vdw_radii = (
            feats["ref_element"].float() @ vdw_radii.unsqueeze(-1)
        ).squeeze(-1)[0]

        pair_index = torch.triu_indices(
            atom_chain_id.shape[0],
            atom_chain_id.shape[0],
            1,
            device=atom_chain_id.device,
        )

        pair_pad_mask = atom_pad_mask[pair_index].all(dim=0)
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]

        num_chains = atom_chain_id.max() + 1
        connected_chain_index = feats["connected_chain_index"][0]
        connected_chain_matrix = torch.eye(
            num_chains, device=atom_chain_id.device, dtype=torch.bool
        )
        connected_chain_matrix[connected_chain_index[0], connected_chain_index[1]] = (
            True
        )
        connected_chain_matrix[connected_chain_index[1], connected_chain_index[0]] = (
            True
        )
        connected_chain_mask = connected_chain_matrix[
            atom_chain_id[pair_index[0]], atom_chain_id[pair_index[1]]
        ]

        pair_index = pair_index[
            :, pair_pad_mask * pair_ion_mask * ~connected_chain_mask
        ]

        lower_bounds = atom_vdw_radii[pair_index].sum(dim=0) * (
            1.0 - parameters["buffer"]
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class SymmetricChainCOMPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = chain_sizes > 1

        pair_index = feats["symmetric_chain_index"][0]
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]
        pair_index = pair_index[:, pair_ion_mask]
        lower_bounds = torch.full(
            (pair_index.shape[1],),
            parameters["buffer"],
            dtype=torch.float32,
            device=pair_index.device,
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return (
            pair_index,
            (k, lower_bounds, upper_bounds),
            (atom_chain_id, atom_pad_mask),
            None,
            None,
        )


class StereoBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        stereo_bond_index = feats["stereo_bond_index"][0]
        stereo_bond_orientations = feats["stereo_bond_orientations"][0].bool()

        lower_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        upper_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        lower_bounds[stereo_bond_orientations] = torch.pi - parameters["buffer"]
        upper_bounds[stereo_bond_orientations] = float("inf")
        lower_bounds[~stereo_bond_orientations] = float("-inf")
        upper_bounds[~stereo_bond_orientations] = parameters["buffer"]

        k = torch.ones_like(lower_bounds)

        return stereo_bond_index, (k, lower_bounds, upper_bounds), None, None, None


class ChiralAtomPotential(FlatBottomPotential, DihedralPotential):
    def compute_args(self, feats, parameters):
        chiral_atom_index = feats["chiral_atom_index"][0]
        chiral_atom_orientations = feats["chiral_atom_orientations"][0].bool()

        lower_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        upper_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        lower_bounds[chiral_atom_orientations] = parameters["buffer"]
        upper_bounds[chiral_atom_orientations] = float("inf")
        upper_bounds[~chiral_atom_orientations] = -1 * parameters["buffer"]
        lower_bounds[~chiral_atom_orientations] = float("-inf")

        k = torch.ones_like(lower_bounds)
        return chiral_atom_index, (k, lower_bounds, upper_bounds), None, None, None


class PlanarBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        double_bond_index = feats["planar_bond_index"][0].T
        double_bond_improper_index = torch.tensor(
            [
                [1, 2, 3, 0],
                [4, 5, 0, 3],
            ],
            device=double_bond_index.device,
        ).T
        improper_index = (
            double_bond_index[:, double_bond_improper_index]
            .swapaxes(0, 1)
            .flatten(start_dim=1)
        )
        lower_bounds = None
        upper_bounds = torch.full(
            (improper_index.shape[1],),
            parameters["buffer"],
            device=improper_index.device,
        )
        k = torch.ones_like(upper_bounds)

        return improper_index, (k, lower_bounds, upper_bounds), None, None, None


class TemplateReferencePotential(FlatBottomPotential, ReferencePotential):
    def compute_args(self, feats, parameters):
        if "template_mask_cb" not in feats or "template_force" not in feats:
            return torch.empty([1, 0]), None, None, None, None

        template_mask = feats["template_mask_cb"][feats["template_force"]]
        if template_mask.shape[0] == 0:
            return torch.empty([1, 0]), None, None, None, None

        ref_coords = feats["template_cb"][feats["template_force"]].clone()
        ref_mask = feats["template_mask_cb"][feats["template_force"]].clone()
        ref_atom_index = (
            torch.bmm(
                feats["token_to_rep_atom"].float(),
                torch.arange(
                    feats["atom_pad_mask"].shape[1],
                    device=feats["atom_pad_mask"].device,
                    dtype=torch.float32,
                )[None, :, None],
            )
            .squeeze(-1)
            .long()
        )[0]
        ref_token_index = (
            torch.bmm(
                feats["atom_to_token"].float(),
                feats["token_index"].unsqueeze(-1).float(),
            )
            .squeeze(-1)
            .long()
        )[0]

        index = torch.arange(
            template_mask.shape[-1], dtype=torch.long, device=template_mask.device
        )[None]
        upper_bounds = torch.full(
            template_mask.shape, float("inf"), device=index.device, dtype=torch.float32
        )
        ref_idxs = torch.argwhere(template_mask).T
        upper_bounds[ref_idxs.unbind()] = feats["template_force_threshold"][
            feats["template_force"]
        ][ref_idxs[0]]

        lower_bounds = None
        k = torch.ones_like(upper_bounds)
        return (
            index,
            (k, lower_bounds, upper_bounds),
            None,
            (ref_coords, ref_mask, ref_atom_index, ref_token_index),
            None,
        )


class ContactPotentital(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        index = feats["contact_pair_index"][0]
        union_index = feats["contact_union_index"][0]
        negation_mask = feats["contact_negation_mask"][0]
        lower_bounds = None
        upper_bounds = feats["contact_thresholds"][0].clone()
        k = torch.ones_like(upper_bounds)
        return (
            index,
            (k, lower_bounds, upper_bounds),
            None,
            None,
            (negation_mask, union_index),
        )


def get_potentials(steering_args, boltz2=False):
    potentials = []
    if steering_args["fk_steering"] or steering_args["physical_guidance_update"]:
        potentials.extend(
            [
                SymmetricChainCOMPotential(
                    parameters={
                        "guidance_interval": 4,
                        "guidance_weight": 0.5
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 0.5,
                        "buffer": ExponentialInterpolation(
                            start=1.0, end=5.0, alpha=-2.0
                        ),
                    }
                ),
                VDWOverlapPotential(
                    parameters={
                        "guidance_interval": 5,
                        "guidance_weight": (
                            PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0])
                            if steering_args["physical_guidance_update"]
                            else 0.0
                        ),
                        "resampling_weight": PiecewiseStepFunction(
                            thresholds=[0.6], values=[0.01, 0.0]
                        ),
                        "buffer": 0.225,
                    }
                ),
                ConnectionsPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.15
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 2.0,
                    }
                ),
                PoseBustersPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.01
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 0.1,
                        "bond_buffer": 0.125,
                        "angle_buffer": 0.125,
                        "clash_buffer": 0.10,
                    }
                ),
                ChiralAtomPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.1
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                StereoBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                PlanarBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.26180,
                    }
                ),
            ]
        )
    if boltz2 and (
        steering_args["fk_steering"] or steering_args["contact_guidance_update"]
    ):
        potentials.extend(
            [
                ContactPotentital(
                    parameters={
                        "guidance_interval": 4,
                        "guidance_weight": (
                            PiecewiseStepFunction(
                                thresholds=[0.25, 0.75], values=[0.0, 0.5, 1.0]
                            )
                            if steering_args["contact_guidance_update"]
                            else 0.0
                        ),
                        "resampling_weight": 1.0,
                        "union_lambda": ExponentialInterpolation(
                            start=8.0, end=0.0, alpha=-2.0
                        ),
                    }
                ),
                TemplateReferencePotential(
                    parameters={
                        "guidance_interval": 2,
                        "guidance_weight": 0.1
                        if steering_args["contact_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                    }
                ),
            ]
        )
    return potentials

# ============================================================================
# HYBRID MODULES (can switch between PyTorch and Tenstorrent)
# ============================================================================

class TemplateV2Module(nn.Module):
    """Template module."""

    def __init__(
        self,
        token_z: int,
        template_dim: int,
        template_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        num_bins: int = 38,
        use_tenstorrent: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the template module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.

        """
        super().__init__()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.use_tenstorrent = use_tenstorrent
        self.relu = nn.ReLU()
        self.z_norm = nn.LayerNorm(token_z)
        self.v_norm = nn.LayerNorm(template_dim)
        self.z_proj = nn.Linear(token_z, template_dim, bias=False)
        self.a_proj = nn.Linear(
            const.num_tokens * 2 + num_bins + 5,
            template_dim,
            bias=False,
        )
        self.u_proj = nn.Linear(template_dim, token_z, bias=False)
        _, _, PairformerNoSeqModule_, _ = _get_pytorch_modules()
        self.pairformer = (
            tenstorrent.PairformerModule(2, 32, 4, None, None, False)
            if use_tenstorrent
            else PairformerNoSeqModule_(
                template_dim,
                num_blocks=template_blocks,
                dropout=dropout,
                pairwise_head_width=pairwise_head_width,
                pairwise_num_heads=pairwise_num_heads,
                post_layer_norm=post_layer_norm,
                activation_checkpointing=activation_checkpointing,
            )
        )

    def forward(
        self,
        z: Tensor,
        feats: dict[str, Tensor],
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings
        feats : dict[str, Tensor]
            Input features
        pair_mask : Tensor
            The pair mask

        Returns
        -------
        Tensor
            The updated pairwise embeddings.

        """
        # Load relevant features
        res_type = feats["template_restype"]
        frame_rot = feats["template_frame_rot"]
        frame_t = feats["template_frame_t"]
        frame_mask = feats["template_mask_frame"]
        cb_coords = feats["template_cb"]
        ca_coords = feats["template_ca"]
        cb_mask = feats["template_mask_cb"]
        visibility_ids = feats["visibility_ids"]
        template_mask = feats["template_mask"].any(dim=2).float()
        num_templates = template_mask.sum(dim=1)
        num_templates = num_templates.clamp(min=1)

        # Compute pairwise masks
        b_cb_mask = cb_mask[:, :, :, None] * cb_mask[:, :, None, :]
        b_frame_mask = frame_mask[:, :, :, None] * frame_mask[:, :, None, :]

        b_cb_mask = b_cb_mask[..., None]
        b_frame_mask = b_frame_mask[..., None]

        # Compute asym mask, template features only attend within the same chain
        B, T = res_type.shape[:2]  # noqa: N806
        tmlp_pair_mask = (
            visibility_ids[:, :, :, None] == visibility_ids[:, :, None, :]
        ).float()

        # Compute template features
        with torch.autocast(device_type="cuda", enabled=False):
            # Compute distogram
            cb_dists = torch.cdist(cb_coords, cb_coords)
            boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1)
            boundaries = boundaries.to(cb_dists.device)
            distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
            distogram = one_hot(distogram, num_classes=self.num_bins)

            # Compute unit vector in each frame
            frame_rot = frame_rot.unsqueeze(2).transpose(-1, -2)
            frame_t = frame_t.unsqueeze(2).unsqueeze(-1)
            ca_coords = ca_coords.unsqueeze(3).unsqueeze(-1)
            vector = torch.matmul(frame_rot, (ca_coords - frame_t))
            norm = torch.norm(vector, dim=-1, keepdim=True)
            unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector))
            unit_vector = unit_vector.squeeze(-1)

            # Concatenate input features
            a_tij = [distogram, b_cb_mask, unit_vector, b_frame_mask]
            a_tij = torch.cat(a_tij, dim=-1)
            a_tij = a_tij * tmlp_pair_mask.unsqueeze(-1)

            res_type_i = res_type[:, :, :, None]
            res_type_j = res_type[:, :, None, :]
            res_type_i = res_type_i.expand(-1, -1, -1, res_type.size(2), -1)
            res_type_j = res_type_j.expand(-1, -1, res_type.size(2), -1, -1)
            a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)
            a_tij = self.a_proj(a_tij)

        # Expand mask
        pair_mask = pair_mask[:, None].expand(-1, T, -1, -1)
        pair_mask = pair_mask.reshape(B * T, *pair_mask.shape[2:])

        # Compute input projections
        v = self.z_proj(self.z_norm(z[:, None])) + a_tij
        v = v.view(B * T, *v.shape[2:])
        v = v + (self.pairformer(None, v)[1] if self.use_tenstorrent else self.pairformer(v, pair_mask, use_kernels=use_kernels))
        v = self.v_norm(v)
        v = v.view(B, T, *v.shape[1:])

        # Aggregate templates
        template_mask = template_mask[:, :, None, None, None]
        num_templates = num_templates[:, None, None, None]
        u = (v * template_mask).sum(dim=1) / num_templates.to(v)

        # Compute output projection
        u = self.u_proj(self.relu(u))
        return u


class AtomDiffusion(Module):
    def __init__(
        self,
        score_model_args,
        num_sampling_steps: int = 5,  # number of sampling steps
        sigma_min: float = 0.0004,  # min noise level
        sigma_max: float = 160.0,  # max noise level
        sigma_data: float = 16.0,  # standard deviation of data distribution
        rho: float = 7,  # controls the sampling schedule
        P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std: float = 1.5,  # standard deviation of log-normal distribution from which noise is drawn for training
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        step_scale_random: list = None,
        coordinate_augmentation: bool = True,
        coordinate_augmentation_inference=None,
        compile_score: bool = False,
        alignment_reverse_diff: bool = False,
        synchronize_sigmas: bool = False,
        use_tenstorrent: bool = False,
    ):
        super().__init__()
        self.use_tenstorrent = use_tenstorrent
        DiffusionModule_, _, _, _ = _get_pytorch_modules()
        self.score_model = (
            tenstorrent.DiffusionModule()
            if use_tenstorrent
            else DiffusionModule_(
                **score_model_args,
            )
        )
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )
        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas

        self.token_s = score_model_args["token_s"]
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return self.zero.device

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,  #: Float['b m 3'],
        sigma,  #: Float['b'] | Float[' '] | float,
        network_condition_kwargs: dict,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        r_noisy = self.c_in(padded_sigma) * noised_atom_coords
        times = self.c_noise(sigma)

        r_update = (
            self.score_model(
                r=r_noisy,
                times=times,
                s_inputs=network_condition_kwargs["s_inputs"],
                s_trunk=network_condition_kwargs["s_trunk"],
                q=network_condition_kwargs["diffusion_conditioning"]["q"],
                c=network_condition_kwargs["diffusion_conditioning"]["c"],
                bias_encoder=network_condition_kwargs["diffusion_conditioning"][
                    "atom_enc_bias"
                ],
                bias_token=network_condition_kwargs["diffusion_conditioning"][
                    "token_trans_bias"
                ],
                bias_decoder=network_condition_kwargs["diffusion_conditioning"][
                    "atom_dec_bias"
                ],
                keys_indexing=network_condition_kwargs["diffusion_conditioning"][
                    "to_keys"
                ].keywords["indexing_matrix"],
                mask=network_condition_kwargs["feats"]["atom_pad_mask"],
                atom_to_token=network_condition_kwargs["feats"][
                    "atom_to_token"
                ],
            )
            if self.use_tenstorrent
            else self.score_model(
                r_noisy=r_noisy,
                times=times,
                **network_condition_kwargs,
            )
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * r_update
        )
        return denoised_coords

    def sample_schedule(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        steering_args=None,
        **network_condition_kwargs,
    ):
        if steering_args is not None and (
            steering_args["fk_steering"]
            or steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
        ):
            potentials = get_potentials(steering_args, boltz2=True)

        if steering_args["fk_steering"]:
            multiplicity = multiplicity * steering_args["num_particles"]
            energy_traj = torch.empty((multiplicity, 0), device=self.device)
            resample_weights = torch.ones(multiplicity, device=self.device).reshape(
                -1, steering_args["num_particles"]
            )
        if (
            steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
        ):
            scaled_guidance_update = torch.zeros(
                (multiplicity, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )
        if max_parallel_samples is None:
            max_parallel_samples = multiplicity

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))
        if self.training and self.step_scale_random is not None:
            step_scale = np.random.choice(self.step_scale_random)
        else:
            step_scale = self.step_scale

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        token_repr = None
        atom_coords_denoised = None

        # gradually denoise
        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            random_R, random_tr = compute_random_augmentation(
                multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
            )
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
            atom_coords = (
                torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
            )
            if atom_coords_denoised is not None:
                atom_coords_denoised -= atom_coords_denoised.mean(dim=-2, keepdims=True)
                atom_coords_denoised = (
                    torch.einsum("bmd,bds->bms", atom_coords_denoised, random_R)
                    + random_tr
                )
            if (
                steering_args["physical_guidance_update"]
                or steering_args["contact_guidance_update"]
            ) and scaled_guidance_update is not None:
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms", scaled_guidance_update, random_R
                )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            steering_t = 1.0 - (step_idx / num_sampling_steps)
            noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = sqrt(noise_var) * torch.randn(shape, device=self.device)
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
                sample_ids = torch.arange(multiplicity).to(atom_coords_noisy.device)
                sample_ids_chunks = sample_ids.chunk(
                    multiplicity % max_parallel_samples + 1
                )

                for sample_ids_chunk in sample_ids_chunks:
                    atom_coords_denoised_chunk = self.preconditioned_network_forward(
                        atom_coords_noisy[sample_ids_chunk],
                        t_hat,
                        network_condition_kwargs=dict(
                            multiplicity=sample_ids_chunk.numel(),
                            **network_condition_kwargs,
                        ),
                    )
                    atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk

                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    # Compute energy of x_0 prediction
                    energy = torch.zeros(multiplicity, device=self.device)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        if parameters["resampling_weight"] > 0:
                            component_energy = potential.compute(
                                atom_coords_denoised,
                                network_condition_kwargs["feats"],
                                parameters,
                            )
                            energy += parameters["resampling_weight"] * component_energy
                    energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

                    # Compute log G values
                    if step_idx == 0:
                        log_G = -1 * energy
                    else:
                        log_G = energy_traj[:, -2] - energy_traj[:, -1]

                    # Compute ll difference between guided and unguided transition distribution
                    if (
                        steering_args["physical_guidance_update"]
                        or steering_args["contact_guidance_update"]
                    ) and noise_var > 0:
                        ll_difference = (
                            eps**2 - (eps + scaled_guidance_update) ** 2
                        ).sum(dim=(-1, -2)) / (2 * noise_var)
                    else:
                        ll_difference = torch.zeros_like(energy)

                    # Compute resampling weights
                    resample_weights = F.softmax(
                        (ll_difference + steering_args["fk_lambda"] * log_G).reshape(
                            -1, steering_args["num_particles"]
                        ),
                        dim=1,
                    )

                # Compute guidance update to x_0 prediction
                if (
                    steering_args["physical_guidance_update"]
                    or steering_args["contact_guidance_update"]
                ) and step_idx < num_sampling_steps - 1:
                    guidance_update = torch.zeros_like(atom_coords_denoised)
                    for guidance_step in range(steering_args["num_gd_steps"]):
                        energy_gradient = torch.zeros_like(atom_coords_denoised)
                        for potential in potentials:
                            parameters = potential.compute_parameters(steering_t)
                            if (
                                parameters["guidance_weight"] > 0
                                and (guidance_step) % parameters["guidance_interval"]
                                == 0
                            ):
                                energy_gradient += parameters[
                                    "guidance_weight"
                                ] * potential.compute_gradient(
                                    atom_coords_denoised + guidance_update,
                                    network_condition_kwargs["feats"],
                                    parameters,
                                )
                        guidance_update -= energy_gradient
                    atom_coords_denoised += guidance_update
                    scaled_guidance_update = (
                        guidance_update
                        * -1
                        * self.step_scale
                        * (sigma_t - t_hat)
                        / t_hat
                    )

                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    resample_indices = (
                        torch.multinomial(
                            resample_weights,
                            (
                                resample_weights.shape[1]
                                if step_idx < num_sampling_steps - 1
                                else 1
                            ),
                            replacement=True,
                        )
                        + resample_weights.shape[1]
                        * torch.arange(
                            resample_weights.shape[0], device=resample_weights.device
                        ).unsqueeze(-1)
                    ).flatten()

                    atom_coords = atom_coords[resample_indices]
                    atom_coords_noisy = atom_coords_noisy[resample_indices]
                    atom_mask = atom_mask[resample_indices]
                    if atom_coords_denoised is not None:
                        atom_coords_denoised = atom_coords_denoised[resample_indices]
                    energy_traj = energy_traj[resample_indices]
                    if (
                        steering_args["physical_guidance_update"]
                        or steering_args["contact_guidance_update"]
                    ):
                        scaled_guidance_update = scaled_guidance_update[
                            resample_indices
                        ]
                    if token_repr is not None:
                        token_repr = token_repr[resample_indices]

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        # training diffusion step
        batch_size = feats["coords"].shape[0] // multiplicity

        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        atom_coords = feats["coords"]

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise

        denoised_atom_coords = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "multiplicity": multiplicity,
                "diffusion_conditioning": diffusion_conditioning,
            },
        )

        return {
            "denoised_atom_coords": denoised_atom_coords,
            "sigmas": sigmas,
            "aligned_true_atom_coords": atom_coords,
        }

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
        filter_by_plddt=0.0,
    ):
        with torch.autocast("cuda", enabled=False):
            denoised_atom_coords = out_dict["denoised_atom_coords"].float()
            sigmas = out_dict["sigmas"].float()

            resolved_atom_mask_uni = feats["atom_resolved_mask"].float()

            if filter_by_plddt > 0:
                plddt_mask = feats["plddt"] > filter_by_plddt
                resolved_atom_mask_uni = resolved_atom_mask_uni * plddt_mask.float()

            resolved_atom_mask = resolved_atom_mask_uni.repeat_interleave(
                multiplicity, 0
            )

            align_weights = denoised_atom_coords.new_ones(denoised_atom_coords.shape[:2])
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["mol_type"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )
            atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

            align_weights = (
                align_weights
                * (
                    1
                    + nucleotide_loss_weight
                    * (
                        torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                        + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
                    )
                    + ligand_loss_weight
                    * torch.eq(
                        atom_type_mult, const.chain_type_ids["NONPOLYMER"]
                    ).float()
                ).float()
            )

            atom_coords = out_dict["aligned_true_atom_coords"].float()
            atom_coords_aligned_ground_truth = weighted_rigid_align(
                atom_coords.detach(),
                denoised_atom_coords.detach(),
                align_weights.detach(),
                mask=feats["atom_resolved_mask"]
                .float()
                .repeat_interleave(multiplicity, 0)
                .detach(),
            )

            # Cast back
            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                denoised_atom_coords
            )

            # weighted MSE loss of denoised atom positions
            mse_loss = (
                (denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2
            ).sum(dim=-1)
            mse_loss = torch.sum(
                mse_loss * align_weights * resolved_atom_mask, dim=-1
            ) / (torch.sum(3 * align_weights * resolved_atom_mask, dim=-1) + 1e-5)

            # weight by sigma factor
            loss_weights = self.loss_weight(sigmas)
            mse_loss = (mse_loss * loss_weights).mean()

            total_loss = mse_loss

            # proposed auxiliary smooth lddt loss
            lddt_loss = self.zero
            if add_smooth_lddt_loss:
                lddt_loss = smooth_lddt_loss(
                    denoised_atom_coords,
                    feats["coords"],
                    torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                    coords_mask=resolved_atom_mask_uni,
                    multiplicity=multiplicity,
                )

                total_loss = total_loss + lddt_loss

            loss_breakdown = {
                "mse_loss": mse_loss,
                "smooth_lddt_loss": lddt_loss,
            }

        return {"loss": total_loss, "loss_breakdown": loss_breakdown}


class ConfidenceModule(nn.Module):
    """Algorithm 31"""

    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        num_dist_bins=64,
        token_level_confidence=True,
        max_dist=22,
        add_s_to_z_prod=False,
        add_s_input_to_s=False,
        add_z_input_to_z=False,
        maximum_bond_distance=0,
        bond_type_feature=False,
        confidence_args: dict = None,
        compile_pairformer=False,
        fix_sym_check=False,
        cyclic_pos_enc=False,
        return_latent_feats=False,
        conditioning_cutoff_min=None,
        conditioning_cutoff_max=None,
        use_tenstorrent: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.max_num_atoms_per_token = 23
        self.no_update_s = pairformer_args.get("no_update_s", False)
        boundaries = torch.linspace(2, max_dist, num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)
        init.gating_init_(self.dist_bin_pairwise_embed.weight)
        self.token_level_confidence = token_level_confidence

        self.s_to_z = LinearNoBias(token_s, token_z)
        self.s_to_z_transpose = LinearNoBias(token_s, token_z)
        init.gating_init_(self.s_to_z.weight)
        init.gating_init_(self.s_to_z_transpose.weight)

        self.add_s_to_z_prod = add_s_to_z_prod
        if add_s_to_z_prod:
            self.s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
            self.s_to_z_prod_in2 = LinearNoBias(token_s, token_z)
            self.s_to_z_prod_out = LinearNoBias(token_z, token_z)
            init.gating_init_(self.s_to_z_prod_out.weight)

        self.s_inputs_norm = nn.LayerNorm(token_s)
        if not self.no_update_s:
            self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        self.add_s_input_to_s = add_s_input_to_s
        if add_s_input_to_s:
            self.s_input_to_s = LinearNoBias(token_s, token_s)
            init.gating_init_(self.s_input_to_s.weight)

        self.add_z_input_to_z = add_z_input_to_z
        if add_z_input_to_z:
            self.rel_pos = RelativePositionEncoder(
                token_z, fix_sym_check=fix_sym_check, cyclic_pos_enc=cyclic_pos_enc
            )
            self.token_bonds = nn.Linear(
                1 if maximum_bond_distance == 0 else maximum_bond_distance + 2,
                token_z,
                bias=False,
            )
            self.bond_type_feature = bond_type_feature
            if bond_type_feature:
                self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

            self.contact_conditioning = ContactConditioning(
                token_z=token_z,
                cutoff_min=conditioning_cutoff_min,
                cutoff_max=conditioning_cutoff_max,
            )
        pairformer_args["v2"] = True
        _, PairformerModule_, _, _ = _get_pytorch_modules()
        self.pairformer_stack = (
            tenstorrent.PairformerModule(8, 32, 4, 24, 16, True)
            if use_tenstorrent
            else PairformerModule_(
                token_s,
                token_z,
                **pairformer_args,
            )
        )
        self.return_latent_feats = return_latent_feats

        self.confidence_heads = ConfidenceHeads(
            token_s,
            token_z,
            token_level_confidence=token_level_confidence,
            **confidence_args,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s,  # Float['b n ts']
        z,  # Float['b n n tz']
        x_pred,  # Float['bm m 3']
        feats,
        pred_distogram_logits,
        multiplicity=1,
        run_sequentially=False,
        use_kernels: bool = False,
    ):
        if run_sequentially and multiplicity > 1:
            assert z.shape[0] == 1, "Not supported with batch size > 1"
            out_dicts = []
            for sample_idx in range(multiplicity):
                out_dicts.append(  # noqa: PERF401
                    self.forward(
                        s_inputs,
                        s,
                        z,
                        x_pred[sample_idx : sample_idx + 1],
                        feats,
                        pred_distogram_logits,
                        multiplicity=1,
                        run_sequentially=False,
                        use_kernels=use_kernels,
                    )
                )

            out_dict = {}
            for key in out_dicts[0]:
                if key != "pair_chains_iptm":
                    out_dict[key] = torch.cat([out[key] for out in out_dicts], dim=0)
                else:
                    pair_chains_iptm = {}
                    for chain_idx1 in out_dicts[0][key]:
                        chains_iptm = {}
                        for chain_idx2 in out_dicts[0][key][chain_idx1]:
                            chains_iptm[chain_idx2] = torch.cat(
                                [out[key][chain_idx1][chain_idx2] for out in out_dicts],
                                dim=0,
                            )
                        pair_chains_iptm[chain_idx1] = chains_iptm
                    out_dict[key] = pair_chains_iptm
            return out_dict

        s_inputs = self.s_inputs_norm(s_inputs)
        if not self.no_update_s:
            s = self.s_norm(s)

        if self.add_s_input_to_s:
            s = s + self.s_input_to_s(s_inputs)

        z = self.z_norm(z)

        if self.add_z_input_to_z:
            relative_position_encoding = self.rel_pos(feats)
            z = z + relative_position_encoding
            z = z + self.token_bonds(feats["token_bonds"].float())
            if self.bond_type_feature:
                z = z + self.token_bonds_type(feats["type_bonds"].long())
            z = z + self.contact_conditioning(feats)

        s = s.repeat_interleave(multiplicity, 0)

        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )
        if self.add_s_to_z_prod:
            z = z + self.s_to_z_prod_out(
                self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
                * self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
            )

        z = z.repeat_interleave(multiplicity, 0)
        s_inputs = s_inputs.repeat_interleave(multiplicity, 0)

        token_to_rep_atom = feats["token_to_rep_atom"]
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        else:
            BM, N, _ = x_pred.shape
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        d = torch.cdist(x_pred_repr, x_pred_repr)
        distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        distogram = self.dist_bin_pairwise_embed(distogram)
        z = z + distogram

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        s_t, z_t = self.pairformer_stack(
            s, z, mask=mask, pair_mask=pair_mask, use_kernels=use_kernels
        )

        # AF3 has residual connections, we remove them
        s = s_t
        z = z_t

        out_dict = {}

        if self.return_latent_feats:
            out_dict["s_conf"] = s
            out_dict["z_conf"] = z

        # confidence heads
        out_dict.update(
            self.confidence_heads(
                s=s,
                z=z,
                x_pred=x_pred,
                d=d,
                feats=feats,
                multiplicity=multiplicity,
                pred_distogram_logits=pred_distogram_logits,
            )
        )
        return out_dict


class AffinityModule(nn.Module):
    """Algorithm 31"""

    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        transformer_args: dict,
        num_dist_bins=64,
        max_dist=22,
        use_cross_transformer: bool = False,
        groups: dict = {},
        use_tenstorrent: bool = False,
    ):
        super().__init__()
        self.use_tenstorrent = use_tenstorrent
        boundaries = torch.linspace(2, max_dist, num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)
        init.gating_init_(self.dist_bin_pairwise_embed.weight)

        self.s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
        self.s_to_z_prod_in2 = LinearNoBias(token_s, token_z)

        self.z_norm = nn.LayerNorm(token_z)
        self.z_linear = LinearNoBias(token_z, token_z)

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=2,
        )
        _, _, PairformerNoSeqModule_, _ = _get_pytorch_modules()
        self.pairformer_stack = (
            tenstorrent.PairformerModule(
                pairformer_args["num_blocks"], 32, 4, None, None, False, True
            )
            if use_tenstorrent
            else PairformerNoSeqModule_(token_z, **pairformer_args)
        )
        self.affinity_heads = AffinityHeadsTransformer(
            token_z,
            transformer_args["token_s"],
            transformer_args["num_blocks"],
            transformer_args["num_heads"],
            transformer_args["activation_checkpointing"],
            False,
            groups=groups,
        )

    def forward(
        self,
        s_inputs,
        z,
        x_pred,
        feats,
        multiplicity=1,
        use_kernels=False,
    ):
        z = self.z_linear(self.z_norm(z))
        z = z.repeat_interleave(multiplicity, 0)

        z = (
            z
            + self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
            + self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
        )

        token_to_rep_atom = feats["token_to_rep_atom"]
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        else:
            BM, N, _ = x_pred.shape
            B = BM // multiplicity
            mult = multiplicity
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        d = torch.cdist(x_pred_repr, x_pred_repr)

        distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        distogram = self.dist_bin_pairwise_embed(distogram)

        z = z + self.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=distogram)

        pad_token_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        rec_mask = (feats["mol_type"] == 0).repeat_interleave(multiplicity, 0)
        rec_mask = rec_mask * pad_token_mask
        lig_mask = (
            feats["affinity_token_mask"]
            .repeat_interleave(multiplicity, 0)
            .to(torch.bool)
        )
        lig_mask = lig_mask * pad_token_mask
        cross_pair_mask = (
            lig_mask[:, :, None] * rec_mask[:, None, :]
            + rec_mask[:, :, None] * lig_mask[:, None, :]
            + lig_mask[:, :, None] * lig_mask[:, None, :]
        )
        z = (
            self.pairformer_stack(None, z, pair_mask=cross_pair_mask)[1]
            if self.use_tenstorrent
            else self.pairformer_stack(
                z=z, pair_mask=cross_pair_mask, use_kernels=use_kernels
            )
        )

        out_dict = {}

        # affinity heads
        out_dict.update(
            self.affinity_heads(z=z, feats=feats, multiplicity=multiplicity)
        )

        return out_dict


# ============================================================================
# BOLTZ2 MAIN MODEL
# ============================================================================

class Boltz2(nn.Module):
    """Boltz2 model."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: dict[str, Any] = None,
        validation_args: dict[str, Any] = None,
        embedder_args: dict[str, Any] = None,
        msa_args: dict[str, Any] = None,
        pairformer_args: dict[str, Any] = None,
        score_model_args: dict[str, Any] = None,
        diffusion_process_args: dict[str, Any] = None,
        diffusion_loss_args: dict[str, Any] = None,
        confidence_model_args: Optional[dict[str, Any]] = None,
        affinity_model_args: Optional[dict[str, Any]] = None,
        affinity_model_args1: Optional[dict[str, Any]] = None,
        affinity_model_args2: Optional[dict[str, Any]] = None,
        validators: Any = None,
        num_val_datasets: int = 1,
        atom_feature_dim: int = 128,
        template_args: Optional[dict] = None,
        confidence_prediction: bool = True,
        affinity_prediction: bool = False,
        affinity_ensemble: bool = False,
        affinity_mw_correction: bool = True,
        run_trunk_and_structure: bool = True,
        skip_run_structure: bool = False,
        token_level_confidence: bool = True,
        alpha_pae: float = 0.0,
        structure_prediction_training: bool = False,
        validate_structure: bool = True,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        compile_pairformer: bool = False,
        compile_structure: bool = False,
        compile_confidence: bool = False,
        compile_affinity: bool = False,
        compile_msa: bool = False,
        exclude_ions_from_lddt: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        predict_args: Optional[dict[str, Any]] = None,
        fix_sym_check: bool = False,
        cyclic_pos_enc: bool = False,
        aggregate_distogram: bool = True,
        bond_type_feature: bool = False,
        use_no_atom_char: bool = False,
        no_random_recycling_training: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        steering_args: Optional[dict] = None,
        use_templates: bool = False,
        compile_templates: bool = False,
        predict_bfactor: bool = False,
        log_loss_every_steps: int = 50,
        checkpoint_diffusion_conditioning: bool = False,
        use_templates_v2: bool = False,
        use_kernels: bool = False,
        use_tenstorrent: bool = False,
        trace: bool = False,
    ) -> None:
        super().__init__()
        
        # Store all hyperparameters for checkpoint loading
        self.hparams = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "num_bins": num_bins,
            "training_args": training_args or {},
            "validation_args": validation_args or {},
            "embedder_args": embedder_args or {},
            "msa_args": msa_args or {},
            "pairformer_args": pairformer_args or {},
            "score_model_args": score_model_args or {},
            "diffusion_process_args": diffusion_process_args or {},
            "diffusion_loss_args": diffusion_loss_args or {},
            "confidence_model_args": confidence_model_args,
            "affinity_model_args": affinity_model_args,
            "affinity_model_args1": affinity_model_args1,
            "affinity_model_args2": affinity_model_args2,
            "num_val_datasets": num_val_datasets,
            "atom_feature_dim": atom_feature_dim,
            "template_args": template_args,
            "confidence_prediction": confidence_prediction,
            "affinity_prediction": affinity_prediction,
            "affinity_ensemble": affinity_ensemble,
            "affinity_mw_correction": affinity_mw_correction,
            "run_trunk_and_structure": run_trunk_and_structure,
            "skip_run_structure": skip_run_structure,
            "token_level_confidence": token_level_confidence,
            "alpha_pae": alpha_pae,
            "structure_prediction_training": structure_prediction_training,
            "validate_structure": validate_structure,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "compile_pairformer": compile_pairformer,
            "compile_structure": compile_structure,
            "compile_confidence": compile_confidence,
            "compile_affinity": compile_affinity,
            "compile_msa": compile_msa,
            "exclude_ions_from_lddt": exclude_ions_from_lddt,
            "ema": ema,
            "ema_decay": ema_decay,
            "min_dist": min_dist,
            "max_dist": max_dist,
            "predict_args": predict_args,
            "fix_sym_check": fix_sym_check,
            "cyclic_pos_enc": cyclic_pos_enc,
            "aggregate_distogram": aggregate_distogram,
            "bond_type_feature": bond_type_feature,
            "use_no_atom_char": use_no_atom_char,
            "no_random_recycling_training": no_random_recycling_training,
            "use_atom_backbone_feat": use_atom_backbone_feat,
            "use_residue_feats_atoms": use_residue_feats_atoms,
            "conditioning_cutoff_min": conditioning_cutoff_min,
            "conditioning_cutoff_max": conditioning_cutoff_max,
            "steering_args": steering_args,
            "use_templates": use_templates,
            "compile_templates": compile_templates,
            "predict_bfactor": predict_bfactor,
            "log_loss_every_steps": log_loss_every_steps,
            "checkpoint_diffusion_conditioning": checkpoint_diffusion_conditioning,
            "use_templates_v2": use_templates_v2,
            "use_kernels": use_kernels,
            "use_tenstorrent": use_tenstorrent,
            "trace": trace,
        }
        self.trace = trace

        # Inference configuration
        self.predict_args = predict_args
        self.steering_args = steering_args
        self.use_kernels = use_kernels
        self.use_tenstorrent = use_tenstorrent
        self.bond_type_feature = bond_type_feature
        self.is_pairformer_compiled = False
        self.is_msa_compiled = False
        self.is_template_compiled = False

        # Input embeddings
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            "use_no_atom_char": use_no_atom_char,
            "use_atom_backbone_feat": use_atom_backbone_feat,
            "use_residue_feats_atoms": use_residue_feats_atoms,
            **embedder_args,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)

        self.s_init = nn.Linear(token_s, token_s, bias=False)
        self.z_init_1 = nn.Linear(token_s, token_z, bias=False)
        self.z_init_2 = nn.Linear(token_s, token_z, bias=False)

        self.rel_pos = RelativePositionEncoder(
            token_z, fix_sym_check=fix_sym_check, cyclic_pos_enc=cyclic_pos_enc
        )

        self.token_bonds = nn.Linear(1, token_z, bias=False)
        self.bond_type_feature = bond_type_feature
        if bond_type_feature:
            self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

        self.contact_conditioning = ContactConditioning(
            token_z=token_z,
            cutoff_min=conditioning_cutoff_min,
            cutoff_max=conditioning_cutoff_max,
        )

        # Normalization layers
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        # Recycling projections
        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # Set compile rules
        # Big models hit the default cache limit (8)
        torch._dynamo.config.cache_size_limit = 512  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001

        # Pairwise stack
        self.use_templates = use_templates
        if use_templates:
            if use_templates_v2:
                self.template_module = TemplateV2Module(
                    token_z, use_tenstorrent=use_tenstorrent, **template_args
                )
            else:
                self.template_module = TemplateModule(token_z, **template_args)
            if compile_templates:
                self.is_template_compiled = True
                self.template_module = torch.compile(
                    self.template_module,
                    dynamic=False,
                    fullgraph=False,
                )

        _, PairformerModule_, _, MSAModule_ = _get_pytorch_modules()
        self.msa_module = (
            tenstorrent.MSAModule(
                n_blocks=4,
                avg_head_dim=32,
                avg_n_heads=8,
                tri_att_head_dim=32,
                tri_att_n_heads=4,
            )
            if self.use_tenstorrent
            else MSAModule_(
                token_z=token_z,
                token_s=token_s,
                **msa_args,
            )
        )
        if compile_msa:
            self.is_msa_compiled = True
            self.msa_module = torch.compile(
                self.msa_module,
                dynamic=False,
                fullgraph=False,
            )
        self.pairformer_module = (
            tenstorrent.PairformerModule(64, 32, 4, 24, 16, True)
            if use_tenstorrent
            else PairformerModule_(token_s, token_z, **pairformer_args)
        )
        if compile_pairformer:
            self.is_pairformer_compiled = True
            self.pairformer_module = torch.compile(
                self.pairformer_module,
                dynamic=False,
                fullgraph=False,
            )

        self.checkpoint_diffusion_conditioning = checkpoint_diffusion_conditioning
        self.diffusion_conditioning = DiffusionConditioning(
            token_s=token_s,
            token_z=token_z,
            atom_s=atom_s,
            atom_z=atom_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=score_model_args["atom_encoder_depth"],
            atom_encoder_heads=score_model_args["atom_encoder_heads"],
            token_transformer_depth=score_model_args["token_transformer_depth"],
            token_transformer_heads=score_model_args["token_transformer_heads"],
            atom_decoder_depth=score_model_args["atom_decoder_depth"],
            atom_decoder_heads=score_model_args["atom_decoder_heads"],
            atom_feature_dim=atom_feature_dim,
            conditioning_transition_layers=score_model_args[
                "conditioning_transition_layers"
            ],
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )

        # Output modules
        self.structure_module = AtomDiffusion(
            score_model_args={
                "token_s": token_s,
                "atom_s": atom_s,
                "atoms_per_window_queries": atoms_per_window_queries,
                "atoms_per_window_keys": atoms_per_window_keys,
                **score_model_args,
            },
            compile_score=compile_structure,
            use_tenstorrent=use_tenstorrent,
            **diffusion_process_args,
        )
        self.distogram_module = DistogramModule(
            token_z,
            num_bins,
        )
        self.predict_bfactor = predict_bfactor
        if predict_bfactor:
            self.bfactor_module = BFactorModule(token_s, num_bins)

        self.confidence_prediction = confidence_prediction
        self.affinity_prediction = affinity_prediction
        self.affinity_ensemble = affinity_ensemble
        self.affinity_mw_correction = affinity_mw_correction
        self.run_trunk_and_structure = run_trunk_and_structure
        self.skip_run_structure = skip_run_structure
        self.token_level_confidence = token_level_confidence
        self.alpha_pae = alpha_pae
        self.structure_prediction_training = structure_prediction_training

        if self.confidence_prediction:
            self.confidence_module = ConfidenceModule(
                token_s,
                token_z,
                token_level_confidence=token_level_confidence,
                bond_type_feature=bond_type_feature,
                fix_sym_check=fix_sym_check,
                cyclic_pos_enc=cyclic_pos_enc,
                conditioning_cutoff_min=conditioning_cutoff_min,
                conditioning_cutoff_max=conditioning_cutoff_max,
                use_tenstorrent=use_tenstorrent,
                **confidence_model_args,
            )
            if compile_confidence:
                self.confidence_module = torch.compile(
                    self.confidence_module, dynamic=False, fullgraph=False
                )

        if self.affinity_prediction:
            if self.affinity_ensemble:
                self.affinity_module1 = AffinityModule(
                    token_s,
                    token_z,
                    use_tenstorrent=use_tenstorrent,
                    **affinity_model_args1,
                )
                self.affinity_module2 = AffinityModule(
                    token_s,
                    token_z,
                    use_tenstorrent=use_tenstorrent,
                    **affinity_model_args2,
                )
                if compile_affinity:
                    self.affinity_module1 = torch.compile(
                        self.affinity_module1, dynamic=False, fullgraph=False
                    )
                    self.affinity_module2 = torch.compile(
                        self.affinity_module2, dynamic=False, fullgraph=False
                    )
            else:
                self.affinity_module = AffinityModule(
                    token_s,
                    token_z,
                    use_tenstorrent=use_tenstorrent,
                    **affinity_model_args,
                )
                if compile_affinity:
                    self.affinity_module = torch.compile(
                        self.affinity_module, dynamic=False, fullgraph=False
                    )

        # Remove grad from weights they are not trained for ddp
        if not structure_prediction_training:
            for name, param in self.named_parameters():
                if (
                    name.split(".")[0] not in ["confidence_module", "affinity_module"]
                    and "out_token_feat_update" not in name
                ):
                    param.requires_grad = False

    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        run_confidence_sequentially: bool = False,
    ) -> dict[str, Tensor]:
        if self.trace:
            print("[boltz2] forward: input_embedder")
        
        s_inputs = self.input_embedder(feats)

        # Initialize the sequence embeddings
        s_init = self.s_init(s_inputs)

        # Initialize pairwise embeddings
        z_init = (
            self.z_init_1(s_inputs)[:, :, None]
            + self.z_init_2(s_inputs)[:, None, :]
        )
        relative_position_encoding = self.rel_pos(feats)
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feats["token_bonds"].float())
        if self.bond_type_feature:
            z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + self.contact_conditioning(feats)

        # Perform rounds of the pairwise stack
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        # Compute pairwise mask
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        if self.run_trunk_and_structure:
            for i in range(recycling_steps + 1):
                # Apply recycling
                s = s_init + self.s_recycle(self.s_norm(s))
                z = z_init + self.z_recycle(self.z_norm(z))

                # Compute pairwise stack
                if self.use_templates:
                    if self.trace:
                        print("[boltz2] template_module")
                    if self.is_template_compiled:
                        template_module = self.template_module._orig_mod  # noqa: SLF001
                    else:
                        template_module = self.template_module

                    z = z + template_module(
                        z, feats, pair_mask, use_kernels=self.use_kernels
                    )

                if self.trace:
                    print("[boltz2] msa_module")
                if self.is_msa_compiled:
                    msa_module = self.msa_module._orig_mod  # noqa: SLF001
                else:
                    msa_module = self.msa_module

                z = z + msa_module(
                    z, s_inputs, feats, use_kernels=self.use_kernels
                )

                # Revert to uncompiled version for validation
                if self.trace:
                    print("[boltz2] pairformer_module")
                if self.is_pairformer_compiled:
                    pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
                else:
                    pairformer_module = self.pairformer_module

                s, z = pairformer_module(
                    s,
                    z,
                    mask=mask,
                    pair_mask=pair_mask,
                    use_kernels=self.use_kernels,
                )

        pdistogram = self.distogram_module(z)
        dict_out = {
            "pdistogram": pdistogram,
            "s": s,
            "z": z,
        }

        if self.run_trunk_and_structure and not self.skip_run_structure:
            if self.trace:
                print("[boltz2] diffusion_conditioning")
            q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                self.diffusion_conditioning(
                    s_trunk=s,
                    z_trunk=z,
                    relative_position_encoding=relative_position_encoding,
                    feats=feats,
                )
            )
            diffusion_conditioning = {
                "q": q,
                "c": c,
                "to_keys": to_keys,
                "atom_enc_bias": atom_enc_bias,
                "atom_dec_bias": atom_dec_bias,
                "token_trans_bias": token_trans_bias,
            }

            if self.trace:
                print("[boltz2] structure_module.sample")
            with torch.autocast("cuda", enabled=False):
                struct_out = self.structure_module.sample(
                    s_trunk=s.float(),
                    s_inputs=s_inputs.float(),
                    feats=feats,
                    num_sampling_steps=num_sampling_steps,
                    atom_mask=feats["atom_pad_mask"].float(),
                    multiplicity=diffusion_samples,
                    max_parallel_samples=max_parallel_samples,
                    steering_args=self.steering_args,
                    diffusion_conditioning=diffusion_conditioning,
                )
                dict_out.update(struct_out)

            if self.predict_bfactor:
                pbfactor = self.bfactor_module(s)
                dict_out["pbfactor"] = pbfactor

        if self.confidence_prediction:
            if self.trace:
                print("[boltz2] confidence_module")
            dict_out.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    x_pred=(
                        dict_out["sample_atom_coords"].detach()
                        if not self.skip_run_structure
                        else feats["coords"].repeat_interleave(diffusion_samples, 0)
                    ),
                    feats=feats,
                    pred_distogram_logits=(
                        dict_out["pdistogram"][
                            :, :, :, 0
                        ].detach()  # TODO only implemented for 1 distogram
                    ),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
            )

        if self.affinity_prediction:
            if self.trace:
                print("[boltz2] affinity_module")
            pad_token_mask = feats["token_pad_mask"][0]
            rec_mask = feats["mol_type"][0] == 0
            rec_mask = rec_mask * pad_token_mask
            lig_mask = feats["affinity_token_mask"][0].to(torch.bool)
            lig_mask = lig_mask * pad_token_mask
            cross_pair_mask = (
                lig_mask[:, None] * rec_mask[None, :]
                + rec_mask[:, None] * lig_mask[None, :]
                + lig_mask[:, None] * lig_mask[None, :]
            )
            z_affinity = z * cross_pair_mask[None, :, :, None]

            argsort = torch.argsort(dict_out["iptm"], descending=True)
            best_idx = argsort[0].item()
            coords_affinity = dict_out["sample_atom_coords"].detach()[best_idx][
                None, None
            ]
            s_inputs = self.input_embedder(feats, affinity=True)

            with torch.autocast("cuda", enabled=False):
                if self.affinity_ensemble:
                    dict_out_affinity1 = self.affinity_module1(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )

                    dict_out_affinity1["affinity_probability_binary"] = (
                        torch.nn.functional.sigmoid(
                            dict_out_affinity1["affinity_logits_binary"]
                        )
                    )
                    dict_out_affinity2 = self.affinity_module2(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )
                    dict_out_affinity2["affinity_probability_binary"] = (
                        torch.nn.functional.sigmoid(
                            dict_out_affinity2["affinity_logits_binary"]
                        )
                    )

                    dict_out_affinity_ensemble = {
                        "affinity_pred_value": (
                            dict_out_affinity1["affinity_pred_value"]
                            + dict_out_affinity2["affinity_pred_value"]
                        )
                        / 2,
                        "affinity_probability_binary": (
                            dict_out_affinity1["affinity_probability_binary"]
                            + dict_out_affinity2["affinity_probability_binary"]
                        )
                        / 2,
                    }

                    dict_out_affinity1 = {
                        "affinity_pred_value1": dict_out_affinity1[
                            "affinity_pred_value"
                        ],
                        "affinity_probability_binary1": dict_out_affinity1[
                            "affinity_probability_binary"
                        ],
                    }
                    dict_out_affinity2 = {
                        "affinity_pred_value2": dict_out_affinity2[
                            "affinity_pred_value"
                        ],
                        "affinity_probability_binary2": dict_out_affinity2[
                            "affinity_probability_binary"
                        ],
                    }
                    if self.affinity_mw_correction:
                        model_coef = 1.03525938
                        mw_coef = -0.59992683
                        bias = 2.83288489
                        mw_val = feats["affinity_mw"]
                        # affinity_mw is a float (not batched), handle both float and list/tensor cases
                        mw = (mw_val[0] if hasattr(mw_val, '__getitem__') and not isinstance(mw_val, str) else mw_val) ** 0.3
                        dict_out_affinity_ensemble["affinity_pred_value"] = (
                            model_coef
                            * dict_out_affinity_ensemble["affinity_pred_value"]
                            + mw_coef * mw
                            + bias
                        )

                    dict_out.update(dict_out_affinity_ensemble)
                    dict_out.update(dict_out_affinity1)
                    dict_out.update(dict_out_affinity2)
                else:
                    dict_out_affinity = self.affinity_module(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )
                    dict_out.update(
                        {
                            "affinity_pred_value": dict_out_affinity[
                                "affinity_pred_value"
                            ],
                            "affinity_probability_binary": torch.nn.functional.sigmoid(
                                dict_out_affinity["affinity_logits_binary"]
                            ),
                        }
                    )

        return dict_out

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: str = "cpu",
        strict: bool = True,
        **kwargs,
    ) -> "Boltz2":
        """Load model from a checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        map_location : str
            Device to load the checkpoint to.
        strict : bool
            Whether to strictly enforce that the keys in checkpoint match.
        **kwargs
            Additional arguments to override checkpoint hyperparameters.
            
        Returns
        -------
        Boltz2
            The loaded model.
        """
        import inspect
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        
        # Get hyperparameters from checkpoint
        hparams = checkpoint.get("hyper_parameters", {})
        
        # Override with provided kwargs
        hparams.update(kwargs)
        
        # Get valid parameters for __init__
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        
        # Filter out unexpected arguments
        filtered_hparams = {k: v for k, v in hparams.items() if k in valid_params}
        
        # Create model
        model = cls(**filtered_hparams)
        
        # Load state dict
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Remove 'model.' prefix if present (from Lightning checkpoints)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_state_dict[key[6:]] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict, strict=strict)
        return model

    def predict_step(self, batch: dict[str, Tensor]) -> dict:
        """Run prediction on a single batch.
        
        Parameters
        ----------
        batch : dict[str, Tensor]
            The input batch.
            
        Returns
        -------
        dict
            The prediction dictionary.
        """
        try:
            out = self(
                batch,
                recycling_steps=self.predict_args["recycling_steps"],
                num_sampling_steps=self.predict_args["sampling_steps"],
                diffusion_samples=self.predict_args["diffusion_samples"],
                max_parallel_samples=self.predict_args["max_parallel_samples"],
                run_confidence_sequentially=True,
            )
            pred_dict = {"exception": False}
            if "keys_dict_batch" in self.predict_args:
                for key in self.predict_args["keys_dict_batch"]:
                    pred_dict[key] = batch[key]

            pred_dict["masks"] = batch["atom_pad_mask"]
            pred_dict["token_masks"] = batch["token_pad_mask"]
            pred_dict["s"] = out["s"]
            pred_dict["z"] = out["z"]

            if "keys_dict_out" in self.predict_args:
                for key in self.predict_args["keys_dict_out"]:
                    pred_dict[key] = out[key]
            pred_dict["coords"] = out["sample_atom_coords"]
            if self.confidence_prediction:
                pred_dict["pde"] = out["pde"]
                pred_dict["plddt"] = out["plddt"]
                pred_dict["confidence_score"] = (
                    4 * out["complex_plddt"]
                    + (
                        out["iptm"]
                        if not torch.allclose(
                            out["iptm"], torch.zeros_like(out["iptm"])
                        )
                        else out["ptm"]
                    )
                ) / 5

                pred_dict["complex_plddt"] = out["complex_plddt"]
                pred_dict["complex_iplddt"] = out["complex_iplddt"]
                pred_dict["complex_pde"] = out["complex_pde"]
                pred_dict["complex_ipde"] = out["complex_ipde"]
                if self.alpha_pae > 0:
                    pred_dict["pae"] = out["pae"]
                    pred_dict["ptm"] = out["ptm"]
                    pred_dict["iptm"] = out["iptm"]
                    pred_dict["ligand_iptm"] = out["ligand_iptm"]
                    pred_dict["protein_iptm"] = out["protein_iptm"]
                    pred_dict["pair_chains_iptm"] = out["pair_chains_iptm"]
            if self.affinity_prediction:
                pred_dict["affinity_pred_value"] = out["affinity_pred_value"]
                pred_dict["affinity_probability_binary"] = out[
                    "affinity_probability_binary"
                ]
                if self.affinity_ensemble:
                    pred_dict["affinity_pred_value1"] = out["affinity_pred_value1"]
                    pred_dict["affinity_probability_binary1"] = out[
                        "affinity_probability_binary1"
                    ]
                    pred_dict["affinity_pred_value2"] = out["affinity_pred_value2"]
                    pred_dict["affinity_probability_binary2"] = out[
                        "affinity_probability_binary2"
                    ]
            return pred_dict

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise e

