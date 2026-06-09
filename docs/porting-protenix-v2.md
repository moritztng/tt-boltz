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

## Checkpoint status + real-weight validation (UPDATE)

- protenix-v2.pt (464M) is **403/gated** at its official URL — NOT publicly
  downloadable yet. But the v0.5.0/v1.0.0 base checkpoints ARE public
  (`https://protenix.tos-cn-beijing.volces.com/checkpoint/...`). Downloaded
  `protenix_base_default_v0.5.0.pt` (1.47GB) to /home/ttuser/protenix_ckpt/ —
  same AF3 architecture at the base dims already verified.
- **Real-weight validation PASSES:** the full PairformerBlock, loaded with REAL
  trained block-0 weights from the checkpoint (strict load, missing=0/unexpected=0),
  reproduces the Protenix reference on device at PCC s=1.00000 z=0.99829 — far
  stronger than random-weight parity. Codified as a checkpoint-guarded test
  (tests/test_protenix.py::test_real_weight_pairformer_block).
- The checkpoint key names match the reference module names exactly, so every
  verified module can be real-weight-validated the same way. v2 (larger dims)
  drops into the same dim-parametric modules once its checkpoint is public.

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

## Remaining-work breakdown (after the compute core is verified)

The tractable device parity gates are done (15/15 = the whole token-level core).
What's left, by type:
- **Atom encoder/decoder** — the one big *device* sub-effort left: windowed
  attention (`AtomTransformer`, n_queries=32/n_keys=128) + atom feature/bias
  setup. tt-bio's Boltz-2 `AtomAttentionEncoder` takes q/c/atom_enc_bias
  precomputed (caller's job); Protenix computes them inside. Multi-iteration:
  build the atom featurization, map onto tt-bio's `AtomTransformer`, parity-test.
- **Cheap host code, reused as-is (not device gates):** FourierEmbedding
  (fixed seed-42 buffers, just cos), RelativePositionEncoding featurization,
  input/template/constraint feature prep.
- **Assembly of verified pieces:** DiffusionConditioning (relpos + Fourier +
  transitions), the diffusion sampler loop (reverse diffusion over the verified
  DiT block).
- **Complex:** ConfidenceHead (multiple heads + a folding trunk).
- **Release gate (needs USER):** gated 464M checkpoint -> real-weight end-to-end
  + Cα-RMSD; then robustness, --fast, CLI --model protenix-v2, vendoring, README.

Net: the autonomous loop has cleared the high-value device parity work. The
single biggest unblock now is the **gated checkpoint** (user) for end-to-end.

## Real-weight validation finding: diffusion DiT block residual is bf16-sensitive

Real-weight validation (the public base checkpoint) PASSES for: PairformerBlock
(PCC s=1.0 z=0.998), MSABlock, DistogramHead, and — in isolation — the diffusion
APB (0.99979) and ConditionedTransitionBlock (0.99998).

But the FULL diffusion token-transformer block (`out = attn + CTB(attn,s)`) scored
only ~0.81 on real weights with RANDOM test inputs. Root cause (diagnosed, NOT a
port bug): with real trained weights the CTB has ~30x gain on random inputs
(ff std ~18.8 vs attn std ~0.6), so the block output is dominated by ff and a tiny
bf16 error in attn is amplified. This is a pathological random-inputs-into-real-
weights combo the trained model never sees; the sub-parts are individually
correct on real weights. Flag for end-to-end: watch CTB intermediate precision
(bf16 storage of large activations) when validating real-input end-to-end RMSD;
HiFi4 fp32 accumulation is already on. A meaningful full-block real-weight check
needs realistic (in-distribution) inputs, not random.

## Boundary of autonomous parity work (where this paused)

Verified on device (17 tests: 15 random-weight + 3 real-weight; the diffusion
full-block real-weight check is bf16-sensitive under random inputs — see above):
the ENTIRE token-level compute core (Pairformer trunk, MSA module, diffusion
token-transformer block, AdaLN, CTB, DistogramHead), with real-weight
confirmation (public v0.5.0 checkpoint) on Pairformer/MSA/Distogram.

The remaining work is NO LONGER module parity — it's INTEGRATION:
- **Atom encoder/decoder**: tt-bio's AtomEncoder consumes a full Boltz-2 `feats`
  dict (ref_pos/charge/element/atom_name_chars/atom_to_token/res_type/mol_type ->
  atom single+pair, windowed bias, to_keys). Porting Protenix's atom path means
  replicating an atom-featurization pipeline with contracts that differ from
  Protenix's — integration, multi-session.
- **Full featurization pipeline**: input parsing -> MSA/template/CCD/ligand ->
  the feature dict the model consumes (large host code).
- **Orchestration + sampler + ConfidenceHead**, then end-to-end Cα-RMSD.
- **Release gate (needs user):** the v2 (464M) checkpoint is 403/gated; only the
  base/v1 are public. Real-weight end-to-end on v2 needs that checkpoint.

To resume: re-run `/loop ...` and I'll undertake the atom-featurization
integration; or provide the v2 checkpoint and I'll wire end-to-end. The verified
modules + remaps + reconciliation catalog + real-weight tests are all committed.

## Verified v2 config (from the real 464M checkpoint — for the full-model build)

Inferred from protenix-v2.pt shapes (use these to construct the reference modules
so weights load strict; v2 differs from base/v0.5.0 dims):
- Pairformer: c_z=256 (base 128), c_s=384, c_hidden_mul=256, no_heads_pair=8
  (base 4), c_hidden_pair_att=32, AttentionPairBias n_heads=16. 48 blocks.
- MSA: c_m=128 (base 64), c_z=256, OPM c_hidden=32, pwa c=8/n_heads=8; the MSA
  block's pair_stack uses no_heads_pair=8 (NOT the PairformerBlock default 4 — must
  be constructed explicitly or strict load mismatches). 4 blocks.
- Diffusion token transformer: c_a=768, c_s=384, c_z=256, n_heads=16. 24 blocks.
- Distogram: c_z input 256, no_bins=64.
Validated on real v2 weights (PCC>0.98, strict load): Pairformer block (s=0.99998
z=0.99482) and DistogramHead. Modules are dim-parametric so they accept v2; the
only gotcha is passing the non-default block sub-config (e.g. MSA pair_stack heads).

## End-to-end path: concrete requirements (probed)

Running the Protenix REFERENCE inference (to get real feats + a reference
structure for tt-bio end-to-end validation) needs a dependency + data stack the
parity work didn't (we installed protenix --no-deps + only torch/ml_collections/
einops/optree):
- deps: absl-py (config/model), biotite + biopython + rdkit (data/CCD/parsing),
  modelcif, etc. (the protenix.runner subpackage may also need installing from
  the repo, not just the PyPI wheel).
- data: CCD components file (components.v20240608.cif[.rdkit_mol.pkl]) +
  cluster/release files; MSA inputs.
- the reference targets GPU; CPU-only viability for the full pipeline is unknown
  (some ops/deepspeed may assume CUDA).
This is the featurization-pipeline INTEGRATION — a focused multi-session effort,
not a per-iteration parity gate. The verified, dim-parametric tt-bio modules
(real-weight-validated on base + v2) are ready to receive its feats.

## End-to-end integration progress (in flight)

Deps installed into env/ to make the Protenix REFERENCE importable (all kept the
ttnn torch build intact at 2.11.0+cpu): `absl-py`, `biotite==1.0.1` (pinned —
1.2.0 breaks json_to_feature), `fair-esm` (--no-deps), `modelcif`. Now importing:
protenix.model.protenix, protenix.config.config, and the full data pipeline
(json_to_feature, infer_data_pipeline, ccd, parser, featurizer).

Remaining for a reference forward (then tt-bio end-to-end):
1. `protenix.runner` is NOT in the PyPI wheel (repo-level script). Build the model
   via `protenix.config.config.parse_configs(...)` -> `Protenix(configs)` and a
   minimal orchestration, OR fetch runner/ from the repo.
2. Construct config: default ConfigManager gives BASE dims; do base first
   (v0.5.0) then override to v2 dims (c_z=256, no_heads_pair=8, c_a=768, c_m=128).
3. CCD data file (components.v20240608.cif[.rdkit_mol.pkl]) for featurization
   (download from the public TOS endpoint).
4. feats via json_to_feature for a tiny protein -> Protenix.forward -> coords ->
   compare structure / validate the tt-bio modules in the real pipeline.

## MILESTONE: full Protenix-v2 reference model loads on-box (real weights)

The complete 464.4M Protenix-v2 reference model now instantiates and loads the
real v2 weights with strict=True (missing=0, unexpected=0). Recipe (scripts/
protenix_ref_build.py):
1. Repo checkout for configs/+runner/ (GIT_LFS_SKIP_SMUDGE=1 shallow clone; set
   PROTENIX_SRC). The runner/configs are NOT in the PyPI wheel.
2. Stub protenix.model.layer_norm.layer_norm.FusedLayerNorm with a torch
   LayerNorm-equivalent (the real one JIT-compiles a CUDA ext; no CUDA here).
3. Config = deep_update({**configs_base, "data":data_configs, **inference_configs},
   model_configs["protenix-v2"]) -> parse_configs(fill_required_with_null=True).
4. Protenix(cfg).eval(); load_state_dict(strip 'module.' prefix, strict).
Next: feats via the data pipeline (download CCD components.cif) for a tiny
protein -> model.forward -> reference structure -> validate tt-bio end-to-end.

## MILESTONE: full Protenix-v2 reference forward runs end-to-end (real weights, real feats)

scripts/protenix_ref_forward.py runs the complete v2 reference on CPU:
- tiny single-chain protein (38 res) -> SampleDictToFeatures builds feats offline
  (use_msa=False, use_template=False, esm.enable=False; N_msa=1 dummy MSA row).
- N_token=38, N_atom=275; forward emits coordinate (1,275,3), plddt (1,275,50),
  pae (1,38,38,64), pde, contact_probs (38,38), resolved (1,275,2).
- saved to ~/protenix_ref_out.pkl (pred + input_feature_dict) for tt-bio compare.

### Reference env (robust against the shared ~/.local being mutated by other agents)
Isolated venv at ~/protenix_ref_venv on **Python 3.11** (uv-installed; the v2 repo
needs biotite==1.4.0 which requires py>=3.11). Deps: torch==2.7.1+cpu, biotite==1.4.0,
modelcif==1.4, numpy 2.2.6, ml_collections, einops, optree, rdkit, gemmi, biopython,
scikit-learn, tqdm, dm-tree, plus protenix+fair-esm (--no-deps). CCD files in
~/common/ (components.cif 468MB, components.cif.rdkit_mol.pkl 135MB from public TOS).
Force base['triangle_multiplicative']='torch' and ['triangle_attention']='torch' (the
v2 default is 'cuequivariance', a CUDA-only fused kernel).

## MILESTONE: golden intermediate dataset captured (all 5 major modules)

scripts/protenix_ref_forward.py with DUMP_INTERMEDIATES=1 registers forward hooks
(with_kwargs=True) and saves the first-call I/O of every major module to
~/protenix_ref_out.pkl['intermediates']. This is the per-stage validation harness
for the tt-bio trunk. I/O contracts (tiny protein: 38 tokens, 275 atoms, c_s=384,
c_z=256, c_s_inputs=449, c_l=128):

- input_embedder(feat) -> s_inputs (38,449)
- msa_module(feat, z(38,38,256), s_inputs(38,449)) -> z (38,38,256)
- pairformer_stack(s(38,384), z(38,38,256)) -> (s, z)          [already ported+validated]
- diffusion_module(x_noisy(1,275,3), t_hat_noise_level(1,), feat, s_inputs(38,449),
    s_trunk(38,384), z_trunk=None, pair_z(38,38,256), p_lm(1,9,32,128,16),
    c_l(275,128), chunk_size, inplace_safe, enable_efficient_fusion) -> x(1,275,3)
- confidence_head(feat, s_inputs(38,449), s_trunk(38,384), z_trunk(38,38,256),
    x_pred_coords(1,275,3), ...) -> plddt(1,275,50), pae(1,38,38,64),
    pde(1,38,38,64), resolved(1,275,2)

Remaining tt-bio build order (simplest first, validate each vs golden):
1. InputFeatureEmbedder -> s_inputs (incl. atom-level AtomAttentionEncoder).
2. trunk linears (s_init/z_init) + relative_position_encoding + recycling.
3. DiffusionModule (DiffusionConditioning + AtomAttn enc/dec + DiT) + sampler.
4. ConfidenceHead. 5. wire end-to-end -> Ca-RMSD vs ~/protenix_ref_out.pkl coords.

## STRATEGY: reuse map (Boltz-2 ttnn stack + tenstorrent.py primitives already cover most of v2)

tt-bio already has the full AF3 stack on ttnn (boltz2.py) AND Protenix-MATCHED
primitives in tenstorrent.py (AdaLN, ConditionedTransitionBlock, AttentionPairBias,
PairformerLayer, Transition, OuterProductMean, PairWeightedAveraging, Triangle*) —
the latter are already validated vs the Protenix reference (random + real v2). So
the v2 port is mostly remap + small reconcile, plus a few genuinely-new pieces.

Per-module plan (from a focused Boltz-2-vs-Protenix-v2 diff):
- AtomTransformer: REMAP — thin wrapper over the diffusion transformer; reuse
  tenstorrent.py DiT primitives. NEW: local windowed attention (n_queries=32,
  n_keys=128) over precomputed d_lm/v_lm/pad_info window tensors.
- AtomAttentionEncoder / Decoder: NEEDS-RECONCILE — same AF3 flow; build Protenix
  variants on tenstorrent.py primitives. Encoder: ref-feat linears + atom-pair small
  MLP (5-layer, zeros-final) + AtomTransformer + mean atom->token aggregate. Decoder:
  broadcast token->atom + skip + AtomTransformer + LayerNorm->Linear(3).
- DiffusionTransformer (DiT block): tenstorrent.py AttentionPairBias(atom_level)+
  AdaLN+ConditionedTransitionBlock ALREADY match Protenix (validated; boltz2.py's
  SwiGLU variant is Boltz-2-specific — do NOT use it for v2).
- DiffusionConditioning: REMAP — RelativePositionEncoding + LayerNorm/Linear +
  2x Transition (single & pair) + FourierEmbedding(noise). Protenix Fourier is
  trainable (load weights), unlike Boltz-2's frozen one.
- ConfidenceHead: distance one-hot embed -> 4x PairformerLayer (have it, validated)
  -> PAE/PDE/pLDDT/resolved heads + softmax-weighted aggregation.
- Diffusion sampler: EDM noise schedule (gamma0/gamma_min/noise_scale/step_scale),
  N_step centered denoise loop calling diffusion_module; batch N_sample on sample dim.

Genuinely-new ttnn work (rest is remap/reconcile of validated primitives):
1. local windowed atom attention (32x128 blocks) in AtomTransformer.
2. atom featurization (ref_pos/charge/element/name_chars -> q/c, d_lm/v_lm -> p_lm).
3. EDM diffusion sampler loop. 4. confidence distance-embed + aggregation.

Build order: InputFeatureEmbedder (AtomAttnEncoder has_coords=False -> s_inputs 449)
-> trunk linears/recycle -> DiffusionConditioning -> DiffusionModule(+AtomAttn
has_coords=True + Decoder)+sampler -> ConfidenceHead -> end-to-end Ca-RMSD.

## MILESTONE: standalone InputFeatureEmbedder parity gate (PCC 1.0)

scripts/protenix_ife_parity_gate.py drives JUST the Protenix InputFeatureEmbedder
(input_embedder.* real v2 weights) on the golden feat from ~/protenix_ref_out.pkl
and reproduces the captured golden s_inputs (38,449) exactly: maxabs_err 0, PCC 1.0.
=> the atom encoder can be driven in isolation; golden inputs (incl. windowed
d_lm(9,32,128,3)/v_lm(9,32,128,1)/pad_info) + output are the gate for the tt-bio port.

AtomAttentionEncoder(has_coords=False) flow to port (transformer.py:820-949):
- prepare_cache -> c_l (atom single [275,128]) + p_lm (windowed atom-pair
  [n_blocks=9, n_queries=32, n_keys=128, c_atompair=16]) from ref feats + d_lm/v_lm.
- q_l = c_l.clone(); p_lm += linear_cl(relu(c_l_q)) + linear_cm(relu(c_l_k)); p_lm += small_mlp(p_lm).
- atom_transformer(q_l, c_l, p_lm): DiffusionTransformer cross_attention_mode with
  LOCAL WINDOWED attention (32 queries x 128 keys per block) — the new ttnn primitive.
- a = mean-aggregate(relu(linear_q(q_l)) atom->token) -> [38, c_token=384].
- s_inputs = cat([a, restype(32), profile(32), deletion_mean(1)]) = 449.

## MILESTONE: first v2 ttnn module — atom featurization (c_l, p_lm) on-device

tt_bio/protenix.py AtomFeaturization ports AtomAttentionEncoder.prepare_cache
(has_coords=False): c_l (atom single [275,128]) and windowed p_lm
([9,32,128,16]) from ref features (pure linears + arcsinh + elementwise). Validated
on a p150a vs real v2 golden (PCC c_l 0.999997, p_lm 0.999978).
- Reference extraction: scripts/protenix_extract_atomfeat.py (py3.11 venv) dumps
  ~/protenix_atomfeat_gold.pkl. On-device test tests/test_protenix_atomfeat.py loads
  that pkl + runs ttnn — NEEDS NO protenix in system python3 (robust to shared-env
  churn from other agents). Pattern for all subsequent v2 module tests.
Next: the local windowed atom attention (DiffusionTransformer cross_attention_mode,
32 queries x 128 keys, per-head pair bias) to complete AtomAttentionEncoder.

## SPEC: local windowed atom attention (the core new ttnn primitive)

Golden target: scripts/protenix_extract_atomtx.py -> ~/protenix_atomtx_gold.pkl
(q,c,p [275,128]/[275,128]/[9,32,128,16] -> qout [275,128]; 81 weights, 3 blocks).
Atom transformer = DiffusionTransformer(cross_attention_mode=True, n_blocks=3,
c_a=c_atom=128, c_s=c_atom=128, c_z=c_atompair=16, n_heads=4, d_head=32). Each block:
DiffusionTransformerBlock = AttentionPairBias + ConditionedTransitionBlock, residuals
(attn_out=APB(a,s,z); a1=attn_out+a; out=CTB(a1,s)+a1). s=c (atom single cond).

AttentionPairBias (cross_attention_mode, has_s=True), per block:
- q_norm = AdaLN_a(a, s);  kv_norm = AdaLN_kv(q_norm, s)   [note: kv AdaLN takes q_norm]
- Q=linear_q(q_norm) (+bias); K=linear_k(kv_norm), V=linear_v(kv_norm) (no bias);
  heads=4, d_head=32; scale q by 1/sqrt(32).
- pair bias = linear_nobias_z(layernorm_z(p)) -> [9,32,128,4] -> permute [4,9,32,128].
- WINDOWING: n_blocks=ceil(N/32)=9, q_pad=9*32-275=13, pad_left=48. After left-pad 48,
  block i key window = padded_kv[i*32 : i*32+128] (clean stride-32/width-128 sliding).
  Per-block SDPA: scores = q_blk@k_blk^T/sqrt(32) + pair_bias + pad_bias; pad_bias =
  (mask_trunked-1)*1e9 (mask_trunked [9,32,128] from pad_info = key validity per window).
  softmax -> @v_blk -> [9,4,32,32]; scatter back to [275,4,32].
- gate: o *= sigmoid(linear_g(q_norm)); out = linear_o(merge_heads(o)) (no bias).
- final per-block APB add: a_last = linear_a_last(s) (BiasInit -2.0); attn = out ... 
  (verify: a = sigmoid? in DiT it's out_a=ff+attn; linear_a_last applies in has_s gate)
ttnn plan: SDPA with batch=n_blocks(9), heads=4 + additive mask (pair+pad). Window
gather via left-pad + 9 slices/concat (overlap => unfold, not reshape). Reuse
tenstorrent.py AdaLN + ConditionedTransitionBlock (already Protenix-matched/validated).

## MILESTONE: AdaLN remap validated; atom-transformer reuse map

tt-bio AdaLN reproduces Protenix AdaptiveLayerNorm with remap (s_norm<-layernorm_s,
s_scale<-linear_s, s_bias<-linear_nobias_s); on-device PCC 0.999996 vs v2 reference.
=> tt_bio.protenix.remap_adaln().

KEY REUSE: tt-bio AttentionPairBias ALREADY has atom_level=True windowed attention
(Boltz-2's local atom attention: keys_indexing gather + batched SDPA, ATOM_WINDOW=32,
ATOM_DIM=128). Reconcile deltas for Protenix v2 atom transformer:
- DOUBLE AdaLN: Protenix cross_attention_mode applies layernorm_kv (a 2nd AdaLN) to
  q_norm before deriving kv; tt-bio's atom_level derives kv from a windowed gather of
  the single normed s (no 2nd AdaLN). Must apply AdaLN_kv to q_norm pre-gather.
- weight remap (per block): attention_pair_bias.attention.linear_q/k/v/g/o ->
  proj_q/k/v/g/o; layernorm_z->proj_z.0, linear_nobias_z->proj_z.1; layernorm_a->the
  layer's AdaLN; conditioned_transition_block.{adaln, linear_nobias_a1/a2/b, linear_s}.
- Protenix ConditionedTransitionBlock uses SiLU gate (b=silu(a1)*a2) + sigmoid(linear_s(s))
  *linear_b(b) — matches tt-bio ConditionedTransitionBlock (verify a1/a2/b names).
Golden gate: ~/protenix_atomtx_gold.pkl (q,c,p->qout). Next: wire the 3-block atom
transformer (reuse AttentionPairBias atom_level + AdaLN + CTB, add the 2nd kv AdaLN)
and validate vs golden_qout.

## MILESTONE: full atom transformer validated end-to-end (PCC 0.999998 vs golden)

The complete 3-block Protenix-v2 atom transformer is validated against real v2
golden_qout:
- scripts/protenix_atomtx_torch_ref.py: pure-torch reimpl (only extracted weights,
  no protenix) reproduces golden_qout EXACTLY (PCC 1.0, maxerr 1.5e-8) — confirms the
  algorithm: double AdaLN (layernorm_a then layernorm_kv on q_norm), windowing
  (left-pad 48, stride-32/width-128 sliding, mask_trunked validity -inf), two gates
  (per-head linear_g + output sigmoid(linear_a_last(s))), ConditionedTransitionBlock,
  residuals.
- scripts/protenix_atomtx_ttnn_parity.py: ttnn impl reuses tt-bio AdaLN (remap_adaln)
  + ttnn linears/CTB/gates -> PCC 0.999998 (maxerr 0.016 bf16) vs golden_qout.

REMAINING for release: the windowed-attention core (Q.K^T/softmax/.V over the 9x
[32,128] windows) currently uses a host gather in the parity script. Move it
on-device — reuse tt-bio AttentionPairBias(atom_level) windowing (keys_indexing +
batched SDPA) or ttnn slice/concat + batched matmul. The rest is fully on-device.
With the atom transformer done, AtomAttentionEncoder = featurization (done) + this
+ relu(linear_q) mean-aggregate atom->token -> a, then s_inputs concat.

## MILESTONE: AtomTransformer fully ON-DEVICE (PCC 0.999998, releasable)

tt_bio/protenix.py AtomTransformer is now fully on-device — windowed attention via
ttnn pad+slice+concat (32-multiple offsets are tile-aligned) + batched 4D matmul +
softmax, no host gather. 3 blocks, double AdaLN, both gates, CTB. Validated vs real
v2 golden_qout: PCC 0.999998 (tests/test_protenix_atomtx.py). The hardest v2 module
is DONE.
Next: AtomAttentionEncoder = AtomFeaturization (done) -> AtomTransformer (done) ->
relu(linear_no_bias_q) + mean atom->token aggregate -> a (38,384); then
s_inputs = cat([a, restype, profile, deletion_mean]) (38,449) validated vs golden.

## MILESTONE: full InputFeatureEmbedder atom encoder on-device -> s_inputs (PCC 0.999999)

tt_bio/protenix.py AtomAttentionEncoder is complete + fully on-device:
featurization (c_l,p_lm) -> p_lm augmentation (windowed c_l projections + 5-layer
small_mlp) -> 3-block windowed AtomTransformer -> relu(linear_no_bias_q) + mean
atom->token aggregate (host-built averaging matrix @ on-device) -> a -> concat
[a, restype, profile, deletion_mean] = s_inputs (38,449). Validated vs real v2 golden
s_inputs at PCC 0.999999 (tests/test_protenix_ife.py). Gate:
scripts/protenix_extract_ife.py -> ~/protenix_ife_gold.pkl.
THE TRUNK ENTRY POINT IS DONE. Next up the trunk: s_init=linear(s_inputs);
z_init from s_init + relative_position_encoding + token_bond; recycling
{template(n_blocks may be 0 for v2), msa_module, pairformer_stack} -> s,z. Then
diffusion (conditioning + diffusion_module + EDM sampler) -> coords; confidence head.

## MILESTONE: trunk input (s_init, z_init) on-device (PCC 0.999997)

tt_bio/protenix.py TrunkInput: s_init=linear_sinit(s_inputs); z_init=zinit1(s_init)
broadcast + zinit2(s_init) broadcast + relp_linear(relp) + token_bond(token_bonds).
All LinearNoBias. Validated vs real v2 golden (s_init/z_init PCC 0.999997,
tests/test_protenix_trunkin.py). Constraint embedder omitted (no active constraints
in plain folding feat; add later if constraint inputs used).
Remaining trunk: recycle linears (layernorm_z_cycle, linear_no_bias_z_cycle,
layernorm_s, linear_no_bias_s) + recycle loop over msa_module + pairformer_stack
(both already validated on v2 weights). template_embedder: check n_blocks for v2.
Then diffusion (conditioning + module + EDM sampler) -> coords; confidence head.

## TRUNK STRUCTURE (v2) + remaining roadmap

v2 trunk recompute (get_pairformer_output): N_cycle=10 recycles, each:
  z = z_init + linear_no_bias_z_cycle(layernorm_z_cycle(z))
  z += template_embedder(feat, z)        # n_blocks=2 ACTIVE (runs on dummy template
                                          # feats when use_template=False; golden incl. it)
  z = msa_module(feat, z, s_inputs)       # 4 blocks  [validated on v2]
  s = s_init + linear_no_bias_s(layernorm_s(s))
  s, z = pairformer_stack(s, z)           # 48 blocks [validated on v2]
-> returns s_inputs, s, z (golden s,z captured in ~/protenix_ref_out.pkl intermediates).

DONE on-device (all PCC>0.9999 vs real v2): AtomFeaturization, AtomTransformer
(windowed), AtomAttentionEncoder->s_inputs, TrunkInput->s_init/z_init. Token core
(Pairformer/MSA/distogram) validated on v2 (test_protenix.py).

REMAINING (the bulk):
1. TEMPLATE EMBEDDER (2 blocks, new): triangle mult/attn + transitions over template
   feats -> z update. Reuse tenstorrent.py TriangleMultiplication/Attention/Transition.
2. RECYCLE WIRING: recycle linears (layernorm_z_cycle, linear_no_bias_z_cycle,
   layernorm_s, linear_no_bias_s) + 10-cycle loop over template/msa/pairformer with v2
   weight remaps -> validate trunk output s,z vs golden.
3. DIFFUSION: DiffusionConditioning (relpe + transitions + Fourier(noise) + single cond)
   + DiffusionModule (AtomAttentionEncoder has_coords=True + token DiT 24 blocks +
   AtomAttentionDecoder) + EDM sampler loop (gamma/noise schedule, N_step) -> coords.
   golden diffusion_module I/O captured (x_noisy,t,feat,s_inputs,s_trunk,pair_z,... ->
   x(1,275,3)).
4. CONFIDENCE HEAD: distance one-hot embed -> 4x PairformerLayer -> PAE/PDE/pLDDT/
   resolved + softmax-weighted aggregation. golden I/O captured.
5. END-TO-END: wire full forward -> Ca-RMSD vs ~/protenix_ref_out.pkl coords.
6. RELEASE: --fast block-fp8, CLI --model protenix-v2 + scheduler/worker wiring,
   vendoring (no clones), unified README.
