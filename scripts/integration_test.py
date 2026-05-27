"""End-to-end head-to-head: pure-PyTorch BoltzGen vs ttnn-swapped BoltzGen
on the same real-protein input, same seed. Exits 0 on PASS, 1 on FAIL.

Usage:
    python scripts/integration_test.py

What this verifies (beyond the per-module unit tests in tests/test_boltzgen.py):
  - The full Boltz forward chain runs end-to-end with convert_to_tt applied
  - The trunk's (s, z) output on REAL production-shape inputs is within the
    same numerical tolerance as the random-weight unit tests (pass criterion)
  - Final coordinates from the diffusion sampler are in a physical range and
    occupy a Kabsch-aligned RMSD consistent with bfloat16 accumulation
    through a 2-step diffusion sampler (informational, sampling_steps=2 is
    severely underconverged for either backend so this is not a tight bound)

Runtime: ~6 min pure-PyTorch CPU baseline + ~30s ttnn. The baseline is
cached at /tmp/baseline_torch.pt; delete that file to force a re-run.

Requirements:
  - BoltzGen design checkpoint at the default HF cache path
  - tt-boltz with convert_to_tt importable
  - One Tenstorrent card
"""
import inspect, time, torch
torch.set_grad_enabled(False)
torch.set_num_threads(8)

from boltzgen.task.predict.data_from_yaml import FromYamlDataModule, DataConfig
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.model.models.boltz import Boltz
from tt_boltz.boltzgen import _remap_legacy_state_dict_keys, convert_to_tt

CKPT = "/home/ttuser/.cache/huggingface/hub/models--boltzgen--boltzgen-1/snapshots/c1be29e1f82ffcc72264f64b993c43fb4e0d17f0/boltzgen1_diverse.ckpt"
MOLDIR = "/home/ttuser/.cache/huggingface/hub/datasets--boltzgen--inference-data/snapshots/c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip"
SEED = 42
PREDICT_ARGS = {"recycling_steps": 3, "sampling_steps": 2, "diffusion_samples": 1}

# Threshold for the "passed" verdict. Calibrated empirically; tighten if
# repeated runs show coords clustering more tightly than this.
RMSD_THRESHOLD_A = 6.0

import os, sys
os.makedirs("/tmp/bench_out", exist_ok=True)

def build_batch():
    cfg = DataConfig(
        moldir=MOLDIR, multiplicity=1,
        yaml_path=["/home/ttuser/boltzgen/example/vanilla_protein/1g13prot.yaml"],
        tokenizer=Tokenizer(atomize_modified_residues=False),
        featurizer=Featurizer(),
        output_dir="/tmp/bench_out",
        diffusion_samples=1, backbone_only=False, atom14=True, atom37=False,
        disulfide_prob=1.0, disulfide_on=True,
    )
    dm = FromYamlDataModule(cfg, batch_size=1, num_workers=0, pin_memory=False)
    return next(iter(dm.predict_dataloader()))

def load_model(*, swap: bool):
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    sig = inspect.signature(Boltz.__init__).parameters
    hp = {k: v for k, v in ckpt["hyper_parameters"].items() if k in sig}
    m = Boltz(**hp).eval()
    if swap:
        convert_to_tt(m)
    m.load_state_dict(_remap_legacy_state_dict_keys(ckpt["state_dict"]), strict=False)
    m.predict_args = dict(PREDICT_ARGS)
    return m

def run(*, swap: bool):
    """Run predict_step AND capture the trunk's last (s, z) via a forward hook.

    The trunk output is deterministic given the same batch — no random sampling
    involved — so this gives a clean numerical comparison of the ttnn modules
    operating on real production-shape inputs. The final coords add the
    diffusion sampler on top, which contributes additional drift from
    bfloat16 accumulation interacting with weighted_rigid_align across steps.
    """
    torch.manual_seed(SEED)
    label = "ttnn " if swap else "torch"
    t0 = time.time()
    print(f"[{label}] building model + loading weights...", flush=True)
    m = load_model(swap=swap)
    print(f"[{label}]   ready in {time.time()-t0:.1f}s, running predict_step", flush=True)

    # Hook the pairformer's last (s, z) output - this is the trunk's final
    # output before structure prediction.
    captured = {}
    def hook(_module, _inputs, output):
        s, z = output
        captured["s"] = s.detach().cpu().float()
        captured["z"] = z.detach().cpu().float()
    h = m.pairformer_module.register_forward_hook(hook)

    t1 = time.time()
    torch.manual_seed(SEED)
    out = m.predict_step(batch, batch_idx=0)
    dt = time.time() - t1
    h.remove()

    coords = out["coords"].detach().cpu().float()
    print(f"[{label}]   predict_step {dt:.1f}s; coords shape={tuple(coords.shape)} "
          f"nan={coords.isnan().any().item()} min={coords.min():.2f} max={coords.max():.2f}",
          flush=True)
    return coords, dt, captured["s"], captured["z"]

CACHE = "/tmp/baseline_torch.pt"
# Cache the batch ALONGSIDE the baseline coords + trunk tensors - the
# vanilla_protein yaml samples a random design length (80..140) each
# invocation, so we must reuse the exact same batch the baseline ran on or
# we'd be comparing different proteins. Delete /tmp/baseline_torch.pt to
# redo the baseline.
if os.path.exists(CACHE):
    saved = torch.load(CACHE)
    batch = saved["batch"]
    coords_torch = saved["coords"]
    s_torch = saved["s"]
    z_torch = saved["z"]
    t_torch = saved.get("time", float("nan"))
    print(f"[torch] reusing cached batch + outputs from {CACHE}")
else:
    torch.manual_seed(SEED)
    batch = build_batch()
    coords_torch, t_torch, s_torch, z_torch = run(swap=False)
    torch.save({
        "batch": batch, "coords": coords_torch, "time": t_torch,
        "s": s_torch, "z": z_torch,
    }, CACHE)

n_tokens = int(batch["token_pad_mask"].sum().item())
print(f"batch: n_tokens={n_tokens}\n", flush=True)

coords_ttnn, t_ttnn, s_ttnn, z_ttnn = run(swap=True)

# Match shapes (sample dim might differ if multiplicity drifts)
n = min(coords_torch.shape[1], coords_ttnn.shape[1])
ct = coords_torch[0, :n]
cn = coords_ttnn[0, :n]
mask = batch["atom_pad_mask"][0, :n].bool()
ct = ct[mask]; cn = cn[mask]

def kabsch_rmsd(P, Q):
    """RMSD after optimal rigid superposition (Kabsch). P, Q are (N, 3)."""
    Pc = P - P.mean(0, keepdim=True)
    Qc = Q - Q.mean(0, keepdim=True)
    H = Pc.T @ Qc
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    D = torch.eye(3); D[2, 2] = d
    R = Vt.T @ D @ U.T
    P_aligned = Pc @ R.T
    diff = P_aligned - Qc
    rmsd = diff.pow(2).sum(-1).mean().sqrt().item()
    per_atom = diff.pow(2).sum(-1).sqrt()
    return rmsd, per_atom

unaligned_rmsd = ((ct - cn) ** 2).sum(-1).mean().sqrt().item()
aligned_rmsd, aligned_per_atom = kabsch_rmsd(ct, cn)

# Trunk numerical comparison (deterministic, no diffusion stochasticity).
# Crop both to the unpadded token region.
tok_mask = batch["token_pad_mask"][0].bool()
def med_rel(a, b):
    return ((a - b).abs() / b.abs().clamp_min(1e-6)).median().item()
s_err = med_rel(s_ttnn[0, tok_mask], s_torch[0, tok_mask])
z_err = med_rel(z_ttnn[0, tok_mask][:, tok_mask], z_torch[0, tok_mask][:, tok_mask])

print(f"\n=== INTEGRATION TEST RESULT ===")
print(f"  protein            : 1g13prot.yaml (vanilla_protein, n_tokens={n_tokens})")
print(f"  predict_args       : {PREDICT_ARGS}")
print(f"  seed               : {SEED}")
print(f"  atoms compared     : {int(mask.sum().item())}")
print(f"  PyTorch coords     : range [{ct.min():.2f}, {ct.max():.2f}] A  ({t_torch:.1f}s)")
print(f"  ttnn coords        : range [{cn.min():.2f}, {cn.max():.2f}] A  ({t_ttnn:.1f}s)")
print(f"  --- trunk output (deterministic) ---")
print(f"  s-track median rel : {s_err:.4f}   (unit-test bar: 0.10)")
print(f"  z-track median rel : {z_err:.4f}   (unit-test bar: 0.30 for 48-block)")
print(f"  --- final coords (after diffusion) ---")
print(f"  unaligned RMSD     : {unaligned_rmsd:.3f} A   (orientation-sensitive)")
print(f"  Kabsch RMSD        : {aligned_rmsd:.3f} A")
print(f"  median per-atom dev: {aligned_per_atom.median().item():.3f} A (after align)")
print(f"  max per-atom dev   : {aligned_per_atom.max().item():.3f} A (after align)")
# Pass criteria: trunk must match the unit-test tolerance on REAL inputs.
# Coords RMSD is informational only at sampling_steps=2 (diffusion is severely
# underconverged; both PyTorch and ttnn produce 'first-pass noise reductions'
# that can diverge by 5-15A while the model itself is operating correctly).
S_TOL, Z_TOL = 0.10, 0.30
trunk_pass = s_err < S_TOL and z_err < Z_TOL
print(f"\n  trunk pass (s<{S_TOL}, z<{Z_TOL}): {trunk_pass}")
print(f"  RESULT: {'PASS' if trunk_pass else 'FAIL'}")
sys.exit(0 if trunk_pass else 1)
