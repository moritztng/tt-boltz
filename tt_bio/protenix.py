"""Protenix-v2 (ByteDance AF3 reproduction) modules on Tenstorrent.

Protenix-v2 is the same AF3 family as Boltz-2 (already ported in boltz2.py) and
shares tt_bio.tenstorrent primitives (AttentionPairBias, AdaLN,
ConditionedTransitionBlock, PairformerLayer, Transition). This module adds the
genuinely-new v2 pieces, built component-by-component and validated against the
real v2 reference (see scripts/protenix_*.py and tests/test_protenix.py).

Status: atom featurization (c_l, p_lm) ported + validated on-device (PCC>0.9999).
"""
import torch
import ttnn

from .tenstorrent import Module, CORE_GRID_MAIN


class AtomFeaturization(Module):
    """Protenix AtomAttentionEncoder.prepare_cache (has_coords=False path).

    Builds the per-atom single embedding c_l and the windowed atom-pair embedding
    p_lm from reference features. Pure linears + arcsinh + elementwise — no
    attention. Reference: protenix/model/modules/transformer.py prepare_cache.

      c_l = W_pos(ref_pos) + W_charge(arcsinh(ref_charge)) + W_f([mask|elem|name])
      c_l *= ref_mask
      p_lm = W_d(d_lm)*v_lm*mask_trunked + W_invd(1/(1+sum d_lm^2))*v_lm + W_v(v_lm)
    """

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        # nn.Linear weights are (out, in); _lin/ttnn.linear want (in, out).
        self.w_ref_pos = self.torch_to_tt("linear_no_bias_ref_pos.weight")
        self.w_ref_charge = self.torch_to_tt("linear_no_bias_ref_charge.weight")
        self.w_f = self.torch_to_tt("linear_no_bias_f.weight")
        self.w_d = self.torch_to_tt("linear_no_bias_d.weight")
        self.w_invd = self.torch_to_tt("linear_no_bias_invd.weight")
        self.w_v = self.torch_to_tt("linear_no_bias_v.weight")

    def _lin_nb(self, x, w):
        return ttnn.linear(
            x, w, compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN,
        )

    def c_l(self, ref_pos, ref_charge_asinh, ref_mask, f_in):
        """All inputs are device tensors. ref_charge_asinh is arcsinh(charge)[...,1],
        ref_mask is [...,1], f_in is cat([mask|element|name_chars]) -> [...,449]."""
        c = ttnn.add(self._lin_nb(ref_pos, self.w_ref_pos),
                     self._lin_nb(ref_charge_asinh, self.w_ref_charge))
        c = ttnn.add(c, self._lin_nb(f_in, self.w_f))
        return ttnn.mul(c, ref_mask)

    def p_lm(self, d_lm, v_lm, invd, mask_trunked):
        """Windowed atom-pair embedding. d_lm/v_lm/invd/mask_trunked are flattened
        to [n_blocks*n_queries*n_keys, *] device tensors (last-dim linears)."""
        p = ttnn.mul(ttnn.mul(self._lin_nb(d_lm, self.w_d), v_lm), mask_trunked)
        p = ttnn.add(p, ttnn.mul(self._lin_nb(invd, self.w_invd), v_lm))
        p = ttnn.add(p, self._lin_nb(v_lm, self.w_v))
        return p
