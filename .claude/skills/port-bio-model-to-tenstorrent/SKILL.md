---
name: port-bio-model-to-tenstorrent
description: >-
  Playbook for implementing/porting a new biomolecular model (protein folding,
  structure prediction, protein language model, diffusion structure head, MSA
  encoder, etc.) onto Tenstorrent hardware inside the tt-bio codebase. Use when
  the task is "port model X to ttnn/Tenstorrent", "add a new model to tt-bio",
  "implement <bio model> on TT", or extending an existing port. Encodes the
  non-negotiable requirements (inference-only, accuracy, performance, --fast,
  unification, no redundancy) and the hard-won methodology (ttnn porting
  patterns, warm profiling, sample-dim batching, bf16 load, accuracy validation,
  device discipline). Derived from the ESMFold2 (ESMC-6B + diffusion) and
  Boltz-2 ports.
---

# Porting a bio model to Tenstorrent (tt-bio)

This is the canonical playbook for adding a new model to **tt-bio** (the package
is `tt_bio/`, the CLI is `tt-bio predict ... --model <name>`). Two models live
here as worked examples: **Boltz-2** (`tt_bio/boltz2.py`) and **ESMFold2**
(`tt_bio/esmfold2.py` + `tt_bio/esmfold2_runtime.py`). Make a new model look
exactly like these — symmetric, unified, no parallel "model-X folder with its
own everything".

## Non-negotiable requirements (the bar every model must clear)

1. **Inference-only.** Delete all training code, losses, optimizers, schedulers.
   **No PyTorch Lightning.** The model is eval-only; filter checkpoint hparams to
   the inference signature.
2. **Pure pip dependencies — no `git clone`.** The reference implementation must
   be a normal pip/`pyproject` dependency (e.g. an HF `transformers` fork on
   PyPI/VCS-pinned in `pyproject.toml`), never a sibling working-copy clone the
   code reaches into. Tests may import the reference; runtime must not depend on
   a clone.
3. **Accuracy must not regress.** Validate per-module against a random-weight
   reference (PCC > 0.98) and end-to-end with Cα-RMSD vs a ground-truth
   structure. ttnn is **not** bit-deterministic (~0.05 Å run-to-run) — compare
   distributions, not single runs.
4. **Performance must not regress** — both **load time** and **predict time**.
5. **Simplicity & elegance.** Reuse the shared `Module`/`TorchWrapper`
   primitives; do not duplicate attention/triangle/linear code.
6. **Unification, zero redundancy.** Same inputs/outputs, same scheduler/worker,
   same CLI surface as the existing models. No copy-pasted pipeline.
7. **`--fast` mode (block-fp8).** Wire the model through `set_fast_mode` /
   `_dtype()` so `--fast` runs bfloat8_b where it's safe; keep fp32/bf16 where
   quantization hurts accuracy.
8. **Complete functionality.** Support every real input (all sizes up to the
   project max, e.g. L=1024 without OOM) and every documented argument /
   checkpoint variant. A "fast checkpoint" or MSA path must work if the model
   has one.
9. **Consistent terminal output** in both normal (Rich) mode and `--debug --log`
   mode — identical look to the other models.
10. **Unified, non-redundant README.** Fold the new model into the existing
    `README.md` sections (don't append a redundant parallel section).
11. **Step by step, simplest case first**, testing in between every step.

## Process (do it in this order)

**Phase 0 — Understand three codebases before writing anything.**
- The target model's reference (its forward pass, module tree, checkpoint layout,
  default inference args — find the *real* defaults, not the demo wrapper's).
- The existing tt-bio ttnn patterns (`tt_bio/tenstorrent.py`, `boltz2.py`,
  `esmfold2.py`, `esmfold2_runtime.py`).
- The tt-metal ttnn `examples/` for op-level idioms.

**Phase 1 — Port component-by-component, simplest first.**
- One module at a time (embeddings → encoder block → trunk → heads → sampler).
- For each: build the reference with **random weights**, load the *same*
  `state_dict` into the ttnn module (`load_state_dict(..., strict=False)`),
  compare with `pcc(out, ref) > 0.98`. See `tests/test_esmfold2.py` +
  `tests/esmfold2_reference.py` for the exact idiom; add `tests/<model>_*.py`.
- Don't proceed to the next module until the current one passes.

**Phase 2 — End-to-end on device.** Real weights, the *entire* model on the TT
device (host should only do glue), produce a structure, validate Cα-RMSD vs
ground truth (Kabsch). Only call it working when accuracy is good on device.

**Phase 3 — Robustness.** All input sizes (bucketed), all args/variants, no OOM
at the size ceiling (size-adaptive release of resident tensors if needed).

**Phase 4 — Optimize** (see Performance below): load time first, then predict
time. Accuracy must stay fixed across every optimization.

**Phase 5 — Unify & document.** Route through the shared CLI/worker, prune dead
code, fold into the README.

## Architecture / how to wire it in (grounded in tt_bio)

- **ttnn modules** subclass `Module` (weights → ttnn) or `TorchWrapper` (torch
  forward that calls into ttnn) in `tt_bio/tenstorrent.py`. Use the shared
  helpers — **do not reimplement**: `torch_to_tt(key, transform, dtype)`,
  `_lin(x, w, bias)`, `_split_heads`, `_merge_heads`, `WeightScope`,
  `CORE_GRID_MAIN`, and the shared `TriangleMultiplication` / `TriangleAttention`
  / `AttentionPairBias` blocks.
- **Two integration styles.** Prefer the **vendored-model + built-in flag**
  style (like `boltz2.py`, which builds ttnn at init) when you own/vendor the
  source. The **external-model + post-hoc patch** style (ESMFold2:
  `esmfold2_runtime.py` reassigns submodules to ttnn wrappers via a `_spec` /
  `_components` table) is for an external pip reference you can't modify — it
  loads weights straight to ttnn and never populates the torch modules.
- **CLI:** add the model to the `--model` `click.Choice` in `tt_bio/main.py`
  (`predict`), reusing `--fast`, `--debug`, `--log`, `--device_ids`. Surface
  unsupported-input notes in **yellow** (warning, not error).
- **Scheduler/worker:** dispatch on `cfg["model"]` in `_WorkerState.load` /
  `predict_one` (`tt_bio/worker.py`) — same scheduler, same multi-device
  fan-out, weights resident per device. Do **not** add a separate run loop.
- **I/O unified:** read inputs and write structures/metrics through the same
  paths the other models use; the output contract (coords, plddt, ptm) must
  match so downstream selection/writing is shared.

## Performance methodology (hard-won — don't relearn these)

- **Profile WARM, never cold.** ttnn compiles each program on first use (seconds,
  one-time, cached per process). A cold single-shot per-stage profile
  misattributes that compile cost. Warm up once, then time the steady-state fold.
- **The device is underutilized at typical sequence lengths (L≈100s) at batch 1.**
  Batching the sample/multiplicity dimension is nearly free (measured: a B=4
  diffusion step ≈ 1.20× B=1; B=8 ≈ 1.89×). Run best-of-N / multi-sample as ONE
  batched B=N pass, not N serial calls. **Before batching, verify the per-step
  ops are batch-safe** (per-element reductions/rotations/SVD, no cross-batch
  mixing).
- **Load weights straight to the device dtype.** Load the checkpoint directly to
  bf16 instead of fp32-then-convert — ~2.6× faster load (half the bytes tiled +
  uploaded), bit-identical at default. Gate fp32 only for `--fast`/quantized
  paths that are mantissa-sensitive.
- **Minimize host↔device transfers.** Keep step-invariant tensors resident on
  device across recurrence/sampling loops (pair state, conditioning, RoPE/band
  tables); transfer only the small per-step tensors.
- **Bucketing for variable-length inputs** (bucket size 64, like Boltz-2) to cap
  kernel-compilation count — done *without* accuracy loss (pad + mask; mirror the
  existing bucketing). Don't over-bucket; verify accuracy is unchanged.
- **`--fast` = block-fp8** via `_dtype()`/`set_fast_mode` (bfloat8_b where safe).
- **Measured DEAD ENDS — do not retry:** (a) parallelizing `from_torch` across
  threads (ttnn serializes the device queue → 1.0×); (b) row-major upload +
  on-device `to_layout` tilize (net *slower* than host TILE_LAYOUT in real
  loads). The load floor after the bf16 win is the serial per-weight upload.
- **Always feasibility-microbench an optimization before committing**, then
  re-verify accuracy. Revert anything that isn't clearly worth it.

## Accuracy methodology

- Keep the random-weight **reference modules** around for regression tests
  (`tests/<model>_reference.py`); per-module PCC > 0.98 is the gate.
- End-to-end: Cα-RMSD via **Kabsch** against a ground-truth structure in
  `examples/ground_truth_structures/`.
- Gotcha: the ESM-style predicted `.cif` omits `_atom_site.occupancy`, so
  BioPython's `MMCIFParser` `KeyError`s — parse Cα coords by column
  (`label_atom_id` idx 2, `Cartn_x/y/z` idx 14/15/16) and Kabsch directly.
- **best-of-N selection only helps when the confidence metric is informative.**
  On low-confidence / out-of-distribution targets pLDDT can be flat or
  anti-correlated with RMSD, so best-of-N can pick a worse structure. Validate
  the benefit on a confident (high-pLDDT) target, not a hard one.

## Hardware & process discipline

- **Pin the device** (`TT_VISIBLE_DEVICES` / `--device_ids`) and confirm it's
  **free before launching** (another job holding it shows as a `CHIP_IN_USE`
  lock / hang).
- **NEVER `SIGTERM` a running device job mid-op** — it can crash the driver.
  Wait for it to finish.
- Run/verify on the device the user specifies; don't assume device 0.

## Definition of done (final checklist)

- [ ] Entire model runs on the TT device; host is glue only.
- [ ] Per-module PCC > 0.98 vs reference; end-to-end RMSD good on device.
- [ ] All input sizes (to the ceiling, no OOM) and all args/variants/checkpoints.
- [ ] `--fast` (block-fp8) works; accuracy acceptable.
- [ ] Load time and predict time measured and not regressed (warm numbers).
- [ ] Routed through the shared CLI (`--model <name>`) + scheduler/worker; multi-
      device works; I/O unified with existing models.
- [ ] Terminal output matches existing models in normal and `--debug --log`.
- [ ] No training code, no Lightning, no git-clone runtime dep; dead code pruned.
- [ ] README updated in the existing sections (no redundant parallel section).
- [ ] Tests added under `tests/`; commit messages end with the project trailer.

## Key files to study / mirror

- `tt_bio/tenstorrent.py` — shared ttnn primitives (`Module`, `TorchWrapper`,
  `torch_to_tt`, `_lin`, `set_fast_mode`, `_dtype`, `get_device`, `WeightScope`,
  triangle/attention blocks, bucketing knobs).
- `tt_bio/boltz2.py` — vendored-model + `use_tenstorrent` integration style.
- `tt_bio/esmfold2.py` — ttnn module wrappers for a folding model.
- `tt_bio/esmfold2_runtime.py` — external-model patch style (`_spec`,
  `_components`, `fold_complex`, resident-loop optimization, best-of-N batching).
- `tt_bio/main.py` — `predict` CLI (`--model`, `--fast`, `--debug`, `--log`).
- `tt_bio/worker.py` — per-device worker; model dispatch in `load`/`predict_one`.
- `tests/test_esmfold2.py`, `tests/esmfold2_reference.py` — the PCC test idiom.
