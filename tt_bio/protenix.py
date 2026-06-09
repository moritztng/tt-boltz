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

from .tenstorrent import Module, CORE_GRID_MAIN, WeightScope


def remap_adaln(sd):
    """Protenix AdaptiveLayerNorm -> tt-bio AdaLN weight names. Math is identical:
    sigmoid(linear_s(LN(s)))*LN(a) + linear_nobias_s(LN(s)).  Validated on-device
    (PCC 0.999996 vs the v2 reference, scripts/ check)."""
    return {
        "s_norm.weight": sd["layernorm_s.weight"],
        "s_scale.weight": sd["linear_s.weight"],
        "s_scale.bias": sd["linear_s.bias"],
        "s_bias.weight": sd["linear_nobias_s.weight"],
    }


class AtomTransformer(Module):
    """Protenix AtomTransformer = DiffusionTransformer(cross_attention_mode=True),
    3 blocks, local windowed attention (n_queries=32, n_keys=128). Fully on-device.

    Each block: AttentionPairBias(double AdaLN q/kv, windowed attn w/ pair bias +
    mask_trunked validity, per-head linear_g gate + output sigmoid(linear_a_last(s))
    gate) -> residual -> ConditionedTransitionBlock -> residual. Validated vs the
    real v2 golden_qout (PCC>0.9999). Reference: transformer.py AtomTransformer.
    """
    N_HEADS = 4
    HEAD_DIM = 32
    N_QUERIES = 32
    N_KEYS = 128
    PAD_LEFT = 48  # (n_keys - n_queries) // 2

    def __init__(self, n_blocks, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_blocks = n_blocks
        self._w = {k: v for k, v in self.weights.data.items()}

    def _T(self, key, t=True):
        w = self._w[key]
        return ttnn.from_torch(w.t().contiguous() if t else w, layout=ttnn.TILE_LAYOUT,
                               device=self.device, dtype=ttnn.bfloat16)

    def _lin(self, x, wkey, bkey=None, activation=None):
        return ttnn.linear(x, self._T(wkey), bias=(self._T(bkey, t=False) if bkey else None),
                           compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN, activation=activation)

    def _adaln(self, a, s, pre):
        from .tenstorrent import AdaLN
        sub = {k[len(pre):]: v for k, v in self._w.items() if k.startswith(pre)}
        return AdaLN(False, remap_adaln(sub), self.compute_kernel_config)(a, s)

    def _windows_q(self, x, N, NP):
        H, dh = self.N_HEADS, self.HEAD_DIM
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.pad(x, [[0, 0], [0, NP - N], [0, 0]], 0.0)
        x = ttnn.reshape(x, (NP // self.N_QUERIES, self.N_QUERIES, H, dh))
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _windows_kv(self, x, N, NP):
        H, dh, nq, nk = self.N_HEADS, self.HEAD_DIM, self.N_QUERIES, self.N_KEYS
        nb = NP // nq
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        Lp = self.PAD_LEFT + NP + nk
        x = ttnn.pad(x, [[0, 0], [self.PAD_LEFT, Lp - self.PAD_LEFT - N], [0, 0]], 0.0)
        blocks = [ttnn.slice(x, [0, i * nq, 0], [1, i * nq + nk, H * dh]) for i in range(nb)]
        x = ttnn.concat(blocks, 0)
        x = ttnn.reshape(x, (nb, nk, H, dh))
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _attention(self, q_norm, kv_norm, p, apb, N, NP, pad_bias):
        H, dh = self.N_HEADS, self.HEAD_DIM
        Q = self._lin(q_norm, apb + "attention.linear_q.weight", apb + "attention.linear_q.bias")
        K = self._lin(kv_norm, apb + "attention.linear_k.weight")
        V = self._lin(kv_norm, apb + "attention.linear_v.weight")
        Qb = self._windows_q(Q, N, NP); Kb = self._windows_kv(K, N, NP); Vb = self._windows_kv(V, N, NP)
        # pair bias: LayerNorm(p, weight only) -> linear_nobias_z -> permute to (nb,H,nq,nk)
        z = ttnn.layer_norm(p, weight=self._T(apb + "layernorm_z.weight", t=False), epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        z = self._lin(z, apb + "linear_nobias_z.weight")          # (nb,nq,nk,H)
        z = ttnn.permute(z, (0, 3, 1, 2))                          # (nb,H,nq,nk)
        sc = ttnn.matmul(Qb, ttnn.permute(Kb, (0, 1, 3, 2)), compute_kernel_config=self.compute_kernel_config)
        sc = ttnn.multiply(sc, dh ** -0.5)
        sc = ttnn.add(ttnn.add(sc, z), pad_bias)
        o = ttnn.matmul(ttnn.softmax(sc, dim=-1), Vb, compute_kernel_config=self.compute_kernel_config)
        o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.reshape(o, (NP, H * dh))
        o = ttnn.slice(ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT), [0, 0], [N, H * dh])
        return ttnn.to_layout(o, ttnn.TILE_LAYOUT)

    def _block(self, a, s, p, b, N, NP, pad_bias):
        P = f"diffusion_transformer.blocks.{b}."; apb = P + "attention_pair_bias."
        q_norm = self._adaln(a, s, apb + "layernorm_a.")
        kv_norm = self._adaln(q_norm, s, apb + "layernorm_kv.")
        o = self._attention(q_norm, kv_norm, p, apb, N, NP, pad_bias)
        g = ttnn.linear(q_norm, self._T(apb + "attention.linear_g.weight"),
                        compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN)
        o = ttnn.multiply(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        attn = self._lin(o, apb + "attention.linear_o.weight")
        gate = self._lin(s, apb + "linear_a_last.weight", apb + "linear_a_last.bias")
        attn = ttnn.multiply(attn, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        a1 = ttnn.add(attn, a)
        ctb = P + "conditioned_transition_block."
        an = self._adaln(a1, s, ctb + "adaln.")
        b1 = self._lin(an, ctb + "linear_nobias_a1.weight", activation="silu")
        b2 = self._lin(an, ctb + "linear_nobias_a2.weight")
        out = self._lin(ttnn.multiply(b1, b2), ctb + "linear_nobias_b.weight")
        cg = self._lin(s, ctb + "linear_s.weight", ctb + "linear_s.bias")
        out = ttnn.multiply(out, cg, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        return ttnn.add(out, a1)

    def __call__(self, a, s, p, mask_trunked):
        """a,s: (1,N,c_atom); p: (nb,nq,nk,c_atompair); mask_trunked: (nb,nq,nk) host
        tensor of per-window key validity. Returns (1,N,c_atom)."""
        N = a.shape[1]
        NP = ((N + self.N_QUERIES - 1) // self.N_QUERIES) * self.N_QUERIES
        pad = torch.where(mask_trunked < 0.5, torch.full_like(mask_trunked, -1e9),
                          torch.zeros_like(mask_trunked)).unsqueeze(1)  # (nb,1,nq,nk)
        pad_bias = ttnn.from_torch(pad, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        x = a
        for b in range(self.n_blocks):
            x = self._block(x, s, p, b, N, NP, pad_bias)
        return x


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


class AtomAttentionEncoder(Module):
    """Protenix InputFeatureEmbedder atom encoder (has_coords=False) -> s_inputs.

    featurization (AtomFeaturization) -> p_lm augmentation (windowed c_l projections
    + small_mlp) -> AtomTransformer -> relu(linear_q) + mean atom->token aggregate
    -> a; then s_inputs = cat([a, restype, profile, deletion_mean]) (c_s_inputs=449).
    Validated vs the real v2 golden s_inputs. Reference: transformer.py
    AtomAttentionEncoder.forward + embedders.py InputFeatureEmbedder.forward.
    """
    NQ, NK, PAD_LEFT, C_ATOMPAIR = 32, 128, 48, 16

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.feat = AtomFeaturization(self.weights, compute_kernel_config)
        self.atx = AtomTransformer(3, self.scope("atom_transformer"), compute_kernel_config)
        self._w = {k: v for k, v in self.weights.data.items()}

    def _T(self, key):
        return ttnn.from_torch(self._w[key].t().contiguous(), layout=ttnn.TILE_LAYOUT,
                               device=self.device, dtype=ttnn.bfloat16)

    def _lin(self, x, key):
        return ttnn.linear(x, self._T(key), compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN)

    def _win_q(self, x, N, NP):  # (1,N,C) -> (nb,nq,C)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.pad(x, [[0, 0], [0, NP - N], [0, 0]], 0.0)
        x = ttnn.reshape(x, (NP // self.NQ, self.NQ, x.shape[-1]))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _win_kv(self, x, N, NP):  # (1,N,C) -> (nb,nk,C)
        C = x.shape[-1]; nb = NP // self.NQ; Lp = self.PAD_LEFT + NP + self.NK
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.pad(x, [[0, 0], [self.PAD_LEFT, Lp - self.PAD_LEFT - N], [0, 0]], 0.0)
        blocks = [ttnn.slice(x, [0, i * self.NQ, 0], [1, i * self.NQ + self.NK, C]) for i in range(nb)]
        x = ttnn.concat(blocks, 0)
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _augment_plm(self, p, c_l, N, NP):
        # p: (nb,nq,nk,16); add windowed c_l projections + small_mlp. c_l: (1,N,128).
        clq = ttnn.relu(self._win_q(c_l, N, NP))            # (nb,nq,128)
        clk = ttnn.relu(self._win_kv(c_l, N, NP))           # (nb,nk,128)
        cl = ttnn.unsqueeze(self._lin(clq, "linear_no_bias_cl.weight"), 2)   # (nb,nq,1,16)
        cm = ttnn.unsqueeze(self._lin(clk, "linear_no_bias_cm.weight"), 1)   # (nb,1,nk,16)
        p = ttnn.add(ttnn.add(p, cl), cm)
        m = self._lin(ttnn.relu(p), "small_mlp.1.weight")
        m = self._lin(ttnn.relu(m), "small_mlp.3.weight")
        m = self._lin(ttnn.relu(m), "small_mlp.5.weight")
        return ttnn.add(p, m)

    def __call__(self, ref_pos, ref_charge_asinh, ref_mask, f_in, d_lm, v_lm, invd,
                 mask_trunked, atom_to_token_mean, restype, profile, deletion_mean):
        """All tensors on device except mask_trunked (host, for the attn pad bias) and
        atom_to_token_mean ((N_token,N) host averaging matrix). p_lm built in windowed
        flat form then reshaped to (nb,nq,nk,16)."""
        N = ref_pos.shape[1] if len(ref_pos.shape) == 3 else ref_pos.shape[0]
        NP = ((N + self.NQ - 1) // self.NQ) * self.NQ
        nb = NP // self.NQ
        c_l = self.feat.c_l(ref_pos, ref_charge_asinh, ref_mask, f_in)        # (1,N,128) or (N,128)
        if len(c_l.shape) == 2:
            c_l = ttnn.reshape(c_l, (1, c_l.shape[0], c_l.shape[1]))
        mt_dev = ttnn.from_torch(mask_trunked.reshape(-1, 1), layout=ttnn.TILE_LAYOUT,
                                 device=self.device, dtype=ttnn.bfloat16)
        p_flat = self.feat.p_lm(d_lm, v_lm, invd, mt_dev)                    # (nb*nq*nk,16)
        p = ttnn.reshape(p_flat, (nb, self.NQ, self.NK, self.C_ATOMPAIR))
        p = self._augment_plm(p, c_l, N, NP)
        q_out = self.atx(c_l, c_l, p, mask_trunked.reshape(nb, self.NQ, self.NK))  # (1,N,128)
        q = ttnn.relu(self._lin(q_out, "linear_no_bias_q.weight"))           # (1,N,384)
        q = ttnn.reshape(q, (N, q.shape[-1]))
        a = ttnn.matmul(atom_to_token_mean, q, compute_kernel_config=self.compute_kernel_config,
                        core_grid=CORE_GRID_MAIN)                            # (N_token,384)
        return ttnn.concat([a, restype, profile, deletion_mean], dim=-1)     # (N_token,449)


class TrunkInput(Module):
    """Protenix trunk input construction: s_inputs -> s_init, z_init.
    s_init = linear_sinit(s_inputs); z_init = zinit1(s_init)[:,None] + zinit2(s_init)[None]
    + relative_position_encoding(relp) + token_bond(token_bonds). All LinearNoBias.
    Reference: protenix/model/protenix.py get_pairformer_output (lines 208-226).
    Validated vs real v2 golden (PCC 0.999997). (Constraint embedder omitted: the
    inference feat carries no active constraints for plain folding.)"""

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self._w = {k: v for k, v in self.weights.data.items()}

    def _lin(self, x, key):
        w = ttnn.from_torch(self._w[key].t().contiguous(), layout=ttnn.TILE_LAYOUT,
                            device=self.device, dtype=ttnn.bfloat16)
        return ttnn.linear(x, w, compute_kernel_config=self.compute_kernel_config, core_grid=CORE_GRID_MAIN)

    def __call__(self, s_inputs, relp, token_bonds):
        """s_inputs (N,449); relp (N,N,139); token_bonds (N,N,1). Returns (s_init (N,c_s),
        z_init (N,N,c_z))."""
        N = s_inputs.shape[0]
        s_init = self._lin(s_inputs, "linear_no_bias_sinit.weight")
        cz = self._w["linear_no_bias_zinit1.weight"].shape[0]
        z1 = ttnn.reshape(self._lin(s_init, "linear_no_bias_zinit1.weight"), (N, 1, cz))
        z2 = ttnn.reshape(self._lin(s_init, "linear_no_bias_zinit2.weight"), (1, N, cz))
        z = ttnn.add(z1, z2)
        z = ttnn.add(z, self._lin(relp, "relative_position_encoding.linear_no_bias.weight"))
        z = ttnn.add(z, self._lin(token_bonds, "linear_no_bias_token_bond.weight"))
        return s_init, z
