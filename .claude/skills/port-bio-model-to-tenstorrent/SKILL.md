---
name: port-bio-model-to-tenstorrent
description: >-
  Playbook for implementing/porting a new biomolecular model (protein folding,
  structure prediction, protein language model, diffusion structure head, MSA
  encoder, etc.) onto Tenstorrent hardware inside the tt-bio codebase. Use when
  the task is "port model X to ttnn/Tenstorrent", "add a new model to tt-bio",
  "implement <bio model> on TT", or extending an existing port. Encodes the
  non-negotiable requirements (inference-only, accuracy, performance, --fast,
  unification, no redundancy, vendored pip deps) and the hard-won methodology
  (component-by-component parity porting, on-device residency, warm profiling,
  bit-exact bucketing, sample-dim batching with OOM chunking, bf16 load).
  Distilled from the ESMC-300M/6B + ESMFold2 and Boltz-2 ports (the 58-commit
  ESMFold2 history).
---

# Porting a bio model to Tenstorrent (tt-bio)

Canonical playbook for adding a model to **tt-bio** (package `tt_bio/`, CLI
`tt-bio predict ... --model <name>`). Worked examples in-tree: **Boltz-2**
(`tt_bio/boltz2.py`) and **ESMFold2** (`tt_bio/esmfold2.py` +
`esmfold2_runtime.py`, vendored reference under `tt_bio/_vendor/`). Make a new
model look exactly like these — symmetric, unified, no parallel "model-X folder
with its own everything".

## Non-negotiable requirements (the bar every model must clear)

1. **Inference-only.** Delete all training code, losses, optimizers, schedulers.
   **No PyTorch Lightning.** Filter checkpoint hparams to the inference signature.
2. **Pure pip deps — no `git clone` at runtime.** **Vendor** the small host-side
   reference into `tt_bio/_vendor/<name>/` and depend on stock PyPI wheels (see
   Vendoring below). No sibling clones, no `sys.path` shims, no `ESM_PATH`.
3. **Accuracy must not regress.** Per-module PCC > 0.98 vs a random-weight
   reference; end-to-end Cα-RMSD vs ground truth (sub-Å is the target — ESMFold2
   hit 0.40–0.43 Å on Trp-cage). ttnn is **not** bit-deterministic (~0.05 Å
   run-to-run) — compare distributions, and see the RNG-confound note below.
4. **Performance must not regress** — both **load time** and **predict time**.
5. **Simplicity & elegance.** Reuse shared `Module`/`TorchWrapper` primitives;
   never duplicate attention/triangle/linear code.
6. **Unification, zero redundancy.** Same I/O, scheduler/worker, and CLI as the
   existing models. After it works, *delete* the parallel/unused paths.
7. **`--fast` mode (block-fp8).** Route through `set_fast_mode`/`_dtype()` so
   `--fast` runs bfloat8_b where safe; keep fp32/bf16 where it hurts accuracy.
8. **Complete functionality.** Every input size (to the ceiling, e.g. L=1024, no
   OOM), every argument, every checkpoint variant (e.g. a "fast" checkpoint),
   MSA path if the model has one.
9. **Consistent terminal output** in normal (Rich) and `--debug --log` modes —
   identical look to the other models.
10. **Unified, non-redundant README** — fold into the existing `--model` table /
    sections; don't append a parallel prose section.
11. **Step by step, simplest case first**, testing in between every step.

## Process (the order that worked)

**Phase 0 — Understand three codebases first:** the target reference (forward,
module tree, checkpoint layout, the *real* inference defaults — not a demo
wrapper's), the tt-bio ttnn patterns (`tenstorrent.py`, `boltz2.py`,
`esmfold2*.py`), and tt-metal `examples/`.

**Phase 1 — Start with the SMALLEST sub-model/variant.** ESMFold2 began with a
from-scratch ttnn port of **ESMC-300M** (sequence-only LM) validated to emb/logits
PCC ~0.999 *before* touching the 6B model or the folding head. Get a small thing
bit-faithful, then scale.

**Build the reference harness as "step 0"** (before any ttnn code). Load *only*
the needed reference module file(s) under stub/fake namespace packages (stub
unused deps) to avoid importing the whole framework or mutating the shared env;
pin the arch config from the actual checkpoint keys. Add any missing shared
primitive to `tenstorrent.py` (RoPE was the one tt-bio lacked).

**Phase 2 — Port component-by-component, each "with parity"** (PCC > 0.98 vs the
random-weight reference via `load_state_dict(strict=False)`; idiom in
`tests/test_esmfold2.py` + `tests/esmfold2_reference.py`). The order that worked
for a folding model:
reference harness → folding trunk (triangle-mult) → token diffusion transformer
(DiT) → diffusion conditioning → atom transformer (SWA + 3D RoPE) → atom
encoder/decoder + diffusion module → distogram + confidence heads → inputs
embedder + relpos → MSA encoder (optional) → LM shim + diffusion sampler →
end-to-end orchestration. Don't advance until the current module passes.
Parallelize this across the machine's TT cards — one experiment per free device
(see Hardware discipline) — so parity tests, seeds, and variants run at once.

**Phase 3 — End-to-end on device.** Real weights, *entire* model on device (host
= glue), validate Cα-RMSD vs ground truth. Only "working" when accuracy is good
**on device**.

**Phase 4 — Robustness.** All sizes (bucketed), all args/variants, no OOM at the
ceiling (size-adaptive release of resident tensors).

**Phase 5 — Optimize** (load first, then predict — see Performance). Accuracy
fixed throughout.

**Phase 6 — Unify & document.** Route through the shared CLI/worker, delete dead
code, fold into the README.

## Architecture / wiring (grounded in tt_bio)

- ttnn modules subclass `Module` or `TorchWrapper` in `tenstorrent.py`. Use the
  shared helpers — never reimplement: `torch_to_tt`, `_lin`, `_split_heads`,
  `_merge_heads`, `WeightScope`, `CORE_GRID_MAIN`, and the shared
  `TriangleMultiplication` / `TriangleAttention` / `AttentionPairBias` blocks.
- **Swap only the computationally heavy ops to ttnn; keep cheap host code as-is.**
  Production runs the *reference* confidence head with only its O(L³) folding
  trunk swapped to ttnn — the full ttnn head was built, never paid off, and was
  deleted. Don't port what's cheap; it's simpler and the accuracy is free.
- **Two integration styles:** prefer **vendored-model + built-in flag** (like
  `boltz2.py`, builds ttnn at init). The **external-model + post-hoc patch**
  style (ESMFold2: `esmfold2_runtime.py` reassigns submodules to ttnn wrappers
  via a single declarative `_SPEC` table + one generic `_Adapter`) suits a
  vendored reference whose forward you reuse.
- **CLI:** add to the `--model` `click.Choice` in `main.py` (`predict`), reusing
  `--fast`/`--debug`/`--log`/`--device_ids`. Surface unsupported-input notes in
  **yellow** (warning, not error). Print the same "Done" summary line.
- **Scheduler/worker:** dispatch on `cfg["model"]` in `_WorkerState.load`/
  `predict_one` (`worker.py`) — same scheduler, same multi-device fan-out,
  weights resident per device, MSA unified worker-side with a shared cache. No
  separate run loop.
- **I/O unified:** same input reading and structure/metrics writing; output
  contract (coords, plddt, ptm) matches so selection/writing is shared.

## Vendoring the reference (how "no clones" is actually done)

- Copy the *small* host-side reference (featurization, MSA, mmCIF assembly, the
  model files) into `tt_bio/_vendor/<name>/`; rewrite absolute imports to the
  vendored namespace (`esm.` → `tt_bio._vendor.esm.`).
- Depend on the **exact stock wheel the model targets** (e.g.
  `transformers==4.57.6`), declared in `pyproject.toml`. Declare *every* import
  as a real dep (`huggingface_hub`, `safetensors`, `zstd`, …) — missing ones
  fail only at install/runtime on a clean machine.
- Ship licenses: per-file provenance headers + the upstream LICENSE texts in the
  vendored dirs + a top-level NOTICE; via `package-data`/`license-files`.
- **Packaging gotcha:** make sure `.gitignore` doesn't swallow vendored dirs (a
  generic `msa/` output-cache rule once hid a vendored `msa/` package → "No
  module named …" at runtime) and that `setuptools packages.find` covers them.

## Performance methodology (hard-won — measured numbers in parens)

- **Profile WARM, never cold.** ttnn compiles each program on first use (seconds,
  one-time, cached per process); a cold single-shot per-stage profile
  misattributes it. Warm up, then time steady state.
- **Profile to find the REAL hotspot — it's counterintuitive.** In ESMFold2 the
  O(L³) folding trunk is ~67% of fold time while the **6B-param LM is only ~2%**.
  Optimize the dominant *stage*, not the biggest-*looking* module.
- **Transfers dominate loops, not compute.** The reference re-transfers the big
  pair tensor (L²·256) host↔device every iteration (host-side tile-layout
  conversion in `from_torch`). Two residency patterns fix this:
  - **Resident recurrence loop:** keep the pair state on device across iterations;
    make deterministic-inference-invariant sub-modules compute-once (~32% faster
    trunk). The reference loop won't speed up warm — host tilization isn't a
    cacheable kernel — which *proves* the transfer cost is real.
  - **Resident sampler context:** split conditioning into **step-invariant**
    (`f(z_trunk, relpos)`, compute once + keep resident) vs **per-step** (cheap,
    t-dependent); per step transfer only the tiny noisy coords. Pattern:
    `prepare()/step()/release_cache()`. (Diffusion 30.6 s → 2.5 s, ~12×.)
- **Load straight to bf16** (not fp32-then-convert): ~2.6× faster load,
  bit-identical at default. Gate fp32 only for quantization-sensitive paths.
- **Keep weights resident for batch/folder prediction** (load once, fold many).
- **`--fast` = block-fp8 on the heavy accuracy-safe op.** Default the trunk's
  triangle-mul to bfloat8_b (trunk −40%, total fold −31%, accuracy unchanged).
  **But keep high matmul accumulation fidelity (fp32 accum / HiFi4)** — LoFi
  accumulation measurably degraded RMSD (0.61 → 0.73 Å) for ~0.5 s; the win is
  fp8 *weights*, not lossy accumulation.
- **Bucketing (cap kernel-compile count for variable lengths) — bit-exact ONLY
  where padding is attention-masked.** Bucket the LM/attention length to 64
  bit-exactly: pad ids, additive −inf mask, AND zero padded keys/values so
  exactness doesn't depend on bf16 `exp(-inf)` (verify PCC = 1.0, maxdiff = 0).
  Do **NOT** bucket a dimension that changes a matmul's *contraction size* — in
  bf16 that isn't exact across sizes and shifts RMSD (~0.065 Å); leave those at
  the smaller existing bucket (32). **Don't over-bucket** — bucketing every dim
  was tried and reverted.
- **best-of-N / multi-sample:** the device is underutilized at small L, so run
  the N samples as one batched **B=N** pass (≈ free: B=4 ≈ 1.20× B=1). BUT it
  replicates the pair state to `[N,L,L,c]` in *both* the sampler and the
  confidence head, so high N OOMs past short lengths. **Chunk both stages over
  the sample axis** at a calibrated `B·L²` budget (distinct per-chunk seeds +
  shrink-on-OOM net), and make the budget **grid-aware** (halve on small Wormhole
  grids vs Blackhole); expose an env override. B=1 path stays untouched.
- **Measure every optimization; revert what doesn't pay.** Real reverts: trimul
  matmul-tiling tuning, and complete-everything bucketing. **Measured DEAD ENDS,
  do not retry:** parallel `from_torch` (ttnn serializes the queue → 1.0×); and
  row-major upload + on-device tilize (net *slower* than host TILE_LAYOUT).
- **Feasibility-microbench before committing**, then re-verify accuracy.

## Accuracy methodology

- Keep random-weight **reference modules** for regression (`tests/<model>_*.py`);
  per-module PCC > 0.98 is the gate.
- End-to-end Cα-RMSD via **Kabsch** vs `examples/ground_truth_structures/`.
- **RNG confound — critical.** The reference may draw from the *global* torch RNG
  for things the `seed` arg doesn't cover (e.g. initial pair state), so per-fold
  RMSD across code changes is confounded and unreliable for proving exactness.
  **Use a bit-identical intermediate test** (hidden-state PCC = 1.0 / maxdiff 0),
  not fold RMSD, to prove an optimization is lossless.
- Predicted-cif gotcha: the ESM-style `.cif` omits `_atom_site.occupancy`, so
  BioPython `MMCIFParser` `KeyError`s — parse Cα by column (`label_atom_id`
  idx 2, `Cartn_x/y/z` idx 14/15/16) + Kabsch directly.
- **best-of-N only helps when confidence is informative.** On low-confidence /
  out-of-distribution targets pLDDT can be flat or anti-correlated with RMSD, so
  best-of-N can pick a *worse* structure. Validate the benefit on a confident
  (high-pLDDT) target, not a hard one.

## Simplification (Phase 6 — be radical, but know where to stop)

After it works, delete aggressively (one ESMFold2 cleanup was −354 lines, same
output): parallel host orchestrations nothing uses, standalone ttnn modules never
wired in, leftover fields. Collapse near-identical wrappers into ONE generic
adapter driven by a declarative spec table. **Do NOT** force-merge genuinely
distinct framework pieces just to merge — name what you intentionally leave
separate.

## Hardware & process discipline

- **Use ALL the free devices to parallelize bring-up.** The machine usually has
  several TT cards (e.g. 4 on a Quiet Box — `ls /dev/tenstorrent/`). Bring-up is
  embarrassingly parallel: fan independent experiments across the cards instead
  of serializing on one. Launch each in its **own process pinned to one card**
  with `TT_VISIBLE_DEVICES=<id>` (run them in the background), so per-module
  parity tests, optimization variants, accuracy/seed sweeps, and size sweeps all
  run **concurrently → ~N× faster iteration** on an N-card box. One job per card.
  The same multi-device machinery serves production — `--device_ids` /
  `--num_devices` fans targets across cards via the shared worker.
- **Pin the device** (`TT_VISIBLE_DEVICES`/`--device_ids`); confirm a card is
  **free** before using it (a held card shows as a `CHIP_IN_USE` lock / hang —
  check `tt-smi`). Respect any user constraint to a specific device.
- **NEVER `SIGTERM` a running device job mid-op** — it can crash the driver. Wait
  for it (and for someone else's job to finish) rather than killing it.
- Account for the grid: Wormhole (8×8) has ~55% of Blackhole's L1, so memory
  budgets (chunk sizes, etc.) must be grid-aware (`_IS_SMALL_GRID`).

## Definition of done

- [ ] Entire model on device; host is glue only.
- [ ] Per-module PCC > 0.98; end-to-end RMSD good on device (proven lossless via
      a bit-identical test, not just fold RMSD).
- [ ] All sizes (ceiling, no OOM), all args/variants/checkpoints, MSA if present.
- [ ] `--fast` (block-fp8) works; accuracy acceptable; high matmul accum kept.
- [ ] Load + predict time measured (warm) and not regressed.
- [ ] Routed through shared CLI (`--model <name>`) + scheduler/worker; multi-
      device works; I/O unified; output matches in normal and `--debug --log`.
- [ ] No training code, no Lightning, no runtime clone (reference vendored +
      licensed; deps in `pyproject`; vendored dirs not gitignored).
- [ ] Dead/redundant code pruned; README folded into existing sections.
- [ ] Tests under `tests/`; commit messages end with the project trailer.

## Key files to study / mirror

- `tt_bio/tenstorrent.py` — shared ttnn primitives (`Module`, `TorchWrapper`,
  `torch_to_tt`, `_lin`, `set_fast_mode`, `_dtype`, `get_device`, `WeightScope`,
  triangle/attention blocks, bucketing + `_IS_SMALL_GRID`).
- `tt_bio/boltz2.py` — vendored-model + built-in-flag integration style.
- `tt_bio/esmfold2.py` — ttnn module wrappers for a folding model.
- `tt_bio/esmfold2_runtime.py` — external-model patch style (`_SPEC`, generic
  `_Adapter`, resident trunk loop, resident sampler context, best-of-N chunking).
- `tt_bio/_vendor/` — vendored reference (host fold path + model files + licenses).
- `tt_bio/main.py` — `predict` CLI; `tt_bio/worker.py` — per-device dispatch.
- `tests/test_esmfold2.py`, `tests/esmfold2_reference.py` — the PCC test idiom.
