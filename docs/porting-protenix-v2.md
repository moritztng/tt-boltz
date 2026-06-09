# Porting Protenix-v2 to Tenstorrent (tt-bio)

Working branch: `protenix-v2`. Follows the
`.claude/skills/port-bio-model-to-tenstorrent` playbook. This doc is the skill's
"Phase 0 — understand + scope" artifact and the living plan; update it as phases
complete.

## Verified status & resume guide (read this first)

**Done (on-device, PCC>0.98, 15 parity tests in tests/test_protenix.py):** the
entire TOKEN-LEVEL compute core of Protenix, reproduced by reusing tt-bio
primitives + Protenix-specific assembly:
- TriangleMultiplication (out/in), TriangleAttention (start/end), Transition,
  AttentionPairBias (Pairformer has_s=False) — and the FULL PairformerBlock.
- OuterProductMean, MSAPairWeightedAveraging — and the FULL MSABlock (assembled).
- AdaLN, ConditionedTransitionBlock (assembled), and the FULL diffusion
  token-transformer block (AdaLN-attn + output gate + CTB).
- DistogramHead.

**Reconciliation catalog (the subtle fixes that recur):** weight remaps per
module (see protenix_reference.py); MSA-block pwa uses c=8 (head_dim=8); diffusion
APB layernorm_z has no bias (default zeros); DistogramHead bias x2 (symmetrize
order); reference forwards MUTATE inputs in place -> clone before the reference
call; OPM/pwa composition order differs from Boltz-2 (assemble in Protenix order).

**Remaining (Phase 1 parity, no checkpoint needed):** atom encoder/decoder (map
to Boltz-2 AtomAttentionEncoder/Decoder — biggest, needs atom featurization),
DiffusionConditioning (assemble: relpos + Fourier + transitions), the diffusion
sampler, input/relpos/template/constraint embedders, ConfidenceHead.

**Release gate (Phase 2+, NEEDS USER):** the **gated 464M Protenix-v2 checkpoint**
must be downloaded (user credentials) to load real weights and run end-to-end;
then Cα-RMSD validation, robustness, --fast, CLI --model protenix-v2, vendoring,
README. The compute core being verified does NOT mean releasable — end-to-end
accuracy on real weights is the gate, and it requires the checkpoint.

## What Protenix-v2 is (Phase 0 findings)

- ByteDance's open-source **AlphaFold3 reproduction** (Apache-2.0). Protenix-v2
  released 2026-04-08, **~464M params** (enhanced capacity = larger
  representation dims + expanded params vs v1), improved antibody–antigen and
  ligand plausibility. At 5 seeds it exceeds v1 at 1000 seeds.
- Predicts all-atom complexes (protein / DNA / RNA / ligand / ion) in one run —
  same scope as Boltz-2.
- **`pip install protenix`** (PyPI, latest 0.5.5). Repo: `bytedance/Protenix`.
- **Dependency hazard:** a full `pip install protenix` pulls `torch==2.3.1+cpu`
  (would clobber the ttnn torch build) + deepspeed / wandb / fair-esm (training
  stack). Per the skill (inference-only, no clobber): the reference is installed
  **`--no-deps`** for parity tests, and the slim inference path will be
  **vendored** into `tt_bio/_vendor/protenix/` for runtime (no clone, no
  training deps).

## Architecture map: Protenix-v2 → existing tt-bio modules

Protenix is the **same AF3 graph as Boltz-2**, which is already fully ported in
tt-bio. Top-level `protenix.model.protenix.Protenix` composition and the tt-bio
equivalent to reuse:

| Protenix module (`protenix/model/...`) | blocks | tt-bio equivalent (already exists) |
|---|---|---|
| `InputFeatureEmbedder` + `RelativePositionEncoding` | — | Boltz-2 input embedder + relpos (`boltz2.py`) |
| `TemplateEmbedder` | — | Boltz-2 template path |
| `MSAModule` | 4 | `tenstorrent.MSA` / `MSAModule`, `MSALayer` |
| `ConstraintEmbedder` | — | **new** (Protenix-specific; optional for first pass) |
| `PairformerStack` | **48** | `tenstorrent.Pairformer` / `PairformerModule`, `PairformerLayer` |
| `DiffusionModule` (atom-enc 3, token-transformer 24, atom-dec 3) | 3/24/3 | `tenstorrent.Diffusion` / `DiffusionModule`, `DiffusionTransformer`, `DiffusionConditioning` (`boltz2.py`) |
| `DistogramHead` | — | Boltz-2 distogram head |
| `ConfidenceHead` | — | Boltz-2 confidence head |
| shared blocks: `tri_attention`, triangle-mult, pair/single attention | — | `TriangleMultiplication`, `TriangleAttention`, `AttentionPairBias` (`tenstorrent.py`) |

Block counts (Pairformer 48, MSA 4, diffusion token 24, atom enc/dec 3) are the
AF3/Boltz-2 defaults. **So the port is mostly: map Protenix-v2's config dims +
weight names onto the existing tt-bio modules, then handle the deltas** (v2's
larger dims, the constraint embedder, and any Protenix-vs-Boltz-2 differences in
template/MSA/confidence wiring). This is materially smaller than the ESMFold2
effort, which needed a new LM + diffusion family from scratch.

Deltas to confirm against Boltz-2 (the real porting work):
- Exact v2 dims `c_s` / `c_z` / `c_atom` / `c_atompair` / `c_m` (read from the v2
  checkpoint config — base config defaults are smaller).
- Constraint embedder (atom contact / pocket constraints) — Protenix-specific.
- Template + RNA-MSA handling vs Boltz-2.
- Diffusion noise schedule / sampler details and confidence-head outputs.

## Plan (skill phases)

- **Phase 0 — understand + scope.** ✅ this doc.
- **Phase 1 — port component-by-component, smallest first, each with a
  random-weight reference parity test (PCC > 0.98).** Reference harness =
  import single `protenix.model.modules.*` classes under stub namespaces
  (avoid the training framework / wandb / custom CUDA layer-norm; use the torch
  fallback). Order: a single triangle-mult / Pairformer block → full
  `PairformerStack` → MSA block/module → diffusion transformer block → atom
  encoder/decoder + diffusion module → input embedder + relpos → distogram +
  confidence heads → constraint/template embedders → end-to-end orchestration.
  Each gate: load Protenix weights into the tt-bio module (`strict=False`) and
  check PCC > 0.98. Add `tests/protenix_reference.py` + `tests/test_protenix.py`.
- **Phase 2 — end-to-end on device.** Real v2 checkpoint, entire model on the TT
  device, Cα-RMSD vs a ground-truth structure (sub-Å target).
- **Phase 3 — robustness.** All sizes (bucketed, no OOM at the ceiling), all
  entity types (protein/DNA/RNA/ligand/ion), all args, constraints, seeds.
- **Phase 4 — optimize.** bf16 load; resident pair state across recycles;
  resident step-invariant diffusion context; `--fast` block-fp8 on the O(L³)
  Pairformer triangle-mult (keep fp32 matmul accumulation); bit-exact bucketing
  where attention-masked; best-of-N batched-but-chunked over samples
  (grid-aware budget). Measure warm; revert what doesn't pay.
- **Phase 5/6 — unify + document.** `tt-bio predict --model protenix-v2` routed
  through the shared scheduler/worker; multi-device fan-out; Rich + `--debug
  --log` output; vendor the reference (+ licenses); fold into the README
  `--model` table; prune dead code.

## Requirements (must all hold — from the skill)

Inference-only (no training code, no Lightning) · pip-only / vendored reference
(no runtime clone) · accuracy preserved (PCC>0.98 + RMSD, proven lossless via a
bit-identical test, not fold RMSD) · load + predict time not regressed · reuse
shared primitives (no duplication) · unified with Boltz-2 (same I/O,
scheduler/worker, CLI) · `--fast` block-fp8 · all sizes/args/entity-types ·
consistent normal + `--debug --log` output · unified non-redundant README ·
step-by-step, simplest first.

## Bring-up acceleration

This machine has multiple TT cards (`ls /dev/tenstorrent/`). Parallelize Phase-1
parity tests / seeds / variants across the **free** cards — one process per card
pinned with `TT_VISIBLE_DEVICES=<id>` — for ~N× faster iteration. One job per
card; confirm free via `tt-smi`; never SIGTERM a running job.

## Status / next steps

- [x] Branch `protenix-v2` created off `main`.
- [x] Phase 0: model researched, reference installed `--no-deps` (+ light deps
      `ml_collections einops optree`), architecture mapped to tt-bio, dependency
      hazard identified, dims confirmed (c_s=384, c_z=128, c_hidden_mul=128,
      pair-att 32, heads 16/4; Pairformer 48, MSA 4, diffusion 24/3/3).
- [x] Phase 1 step 0: `tests/protenix_reference.py` reference harness +
      `tests/test_protenix.py`. **First parity gate PASSES on device:**
      tt-bio `TriangleMultiplication` reproduces Protenix's OpenFold triangle
      multiplication via a weight remap — outgoing PCC **0.99998** (`ending=False`),
      incoming >0.98 (`ending=True`). Confirms the reuse strategy works.
      - Verified remap: tt-bio `g_in`=cat(`linear_a_g`,`linear_b_g`),
        `p_in`=cat(`linear_a_p`,`linear_b_p`), `g_out`=`linear_g`,
        `p_out`=`linear_z`, norms direct; `ending`=incoming.
- [x] TriangleAttention parity on device (PCC>0.98, starting→ending=False,
      ending→ending=True; remap = strip `mha.` prefix). 4/4 tests pass.
- [x] Transition (SwiGLU) parity on device (c_z=128 pair + c_s=384 single,
      PCC>0.98). Remap: norm←layernorm1, fc1←linear_no_bias_a (silu folded in),
      fc2←linear_no_bias_b, fc3←linear_no_bias. **6/6 parity tests pass.**
- [x] **AttentionPairBias (Pairformer, has_s=False) parity on device, PCC 0.99986.**
      Reconciliation: apply the input-`a` LayerNorm externally (tt-bio does it via
      `PairformerLayer.pre_norm_s`); remap `attention.linear_{q,k,v,o,g}`→
      `proj_{q,k,v,g,o}` (q HAS a bias), `layernorm_z`→`proj_z.0`,
      `linear_nobias_z`→`proj_z.1` (raw — tt-bio's internal ×√d is correct as-is).
      **7/7 parity tests pass.** Still TODO: the AdaLN/`has_s=True` variant used by
      the diffusion token transformer.
- [x] **Full PairformerBlock parity on device (both s and z, PCC>0.98).**
      tt-bio's PairformerLayer consumes the block via scopes: tri_mul_out/in
      (fused remap), tri_att_start/end DIRECT (scope strips `mha.`),
      transition_z←pair_transition, attention←attention_pair_bias.attention,
      pre_norm_s←attention_pair_bias.layernorm_a, transition_s←single_transition.
      Residual order matches. **8/8 parity tests pass.** This is the bulk of the
      trunk (×48).
- [x] OuterProductMean (MSA->pair) parity on device, PCC 0.99962 (direct remap:
      norm<-layer_norm, proj_a/b<-linear_1/2, proj_o<-linear_out). 9/9 tests pass.
- [x] MSAPairWeightedAveraging parity on device, PCC>0.98 (norm_m<-layernorm_m,
      proj_m<-linear_no_bias_mv, proj_g<-mg, proj_z<-z, proj_o<-out). 10/10 tests pass.
- [!] **MSA module composition DIVERGES (finding).** Protenix MSABlock order:
      (1) z += OuterProductMean(m)  [OPM on OLD m] -> (2) MSAStack updates m
      (pwa + transition) -> (3) pair_stack (PairformerBlock c_s=0) updates z.
      tt-bio MSALayer order: update m FIRST (pwa+transition) -> THEN OPM(new m)
      -> pairformer_layer. The OPM input differs (old vs new m), so tt-bio's
      MSALayer is NOT a drop-in. ALL MSA sub-modules are verified/reusable
      (OPM, PairWeightedAveraging, Transition, PairformerLayer-on-pair); the MSA
      module port = a small Protenix-ordered assembly reusing those verified
      ttnn primitives. (Pattern: composition layers may diverge from Boltz-2
      even when every primitive matches — check each module's forward order.)
- [~] MSA-layer assembly WIP: OPM-first order composes (OPM runs, output adds
      to z); but PairWeightedAveraging hits a ttnn slice/layout snag on the
      *composed* z (it passes in isolation — a plumbing detail, not architectural).
      Next debug: normalize z's memory_config after the OPM add, or inspect pwa's
      per-head z slicing at L=32. [UPDATE] Root cause was a dims mismatch, not
      plumbing: the MSA-block pwa uses c=8,n_heads=8 (not standalone default c=32) ->
      build PairWeightedAveraging(head_dim=8,n_heads=8). With that, the block RUNS in
      Protenix order (z+=OPM(m); m+=pwa(m,z); m+=transition(m); z=pair_stack(z)) but
      PCC m=0.919 z=0.503 (<0.98). Two isolated correctness gaps to diagnose next:
      (a) head_dim=8 pwa (m~0.92 — likely sub-tile head padding); (b) the c_s=0
      PairformerLayer/transform_s=False path or the OPM-add (z~0.50 — verify the
      c_s=0 pair stack in isolation vs PairformerBlock(c_s=0)). Then MSAModule.
- [ ] Next: build the Protenix-ordered MSA layer (reuse verified primitives) ->
      MSAModule (N blocks + input proj). Reconcile order vs
      Protenix MSABlock) + PairformerStack (N-block) + s/z init linears →
      MSA block/module → diffusion transformer + atom encoder/decoder +
      DiffusionModule → input/relpos/template/constraint embedders →
      distogram + confidence heads. Then real-weight load + end-to-end.
- [ ] Phase 2: download the v2 (464M) checkpoint, pin v2 dims, load real weights,
      end-to-end on device, Cα-RMSD vs ground truth.
- [ ] Phases 3–6: robustness (all entity types/sizes/no-OOM), optimize
      (bf16 load, residency, `--fast`, bucketing, best-of-N chunking), unify CLI
      (`--model protenix-v2` + worker/multi-device + Rich/`--debug --log`),
      vendor the reference, README, prune.

**Not release-ready yet** — this is the start of Phase 1 (the keystone module is
proven on hardware). The remaining modules + real-weight end-to-end + accuracy
+ robustness + CLI/vendoring are required before public Protenix-v2 support.

## Atom encoder/decoder — approach (biggest remaining sub-piece)

Protenix's `AtomAttentionEncoder`/`AtomAttentionDecoder` use AF3 **local
cross-attention** (n_queries=32 / n_keys=128 windowing + `c_atompair=16` pair
bias), NOT 3D-RoPE. So map them onto tt-bio's **Boltz-2** atom modules
(`boltz2.py`: `AtomAttentionEncoder`@1552, `AtomAttentionDecoder`@1633,
`AtomTransformer`@1904) — also AF3 — **not** the esmfold2 SWA (3D-RoPE, wrong
formulation). The parity harness needs full atom featurization (ref_pos / charge
/ element / atom_name_chars / atom_to_token + atompair features); multi-step.
