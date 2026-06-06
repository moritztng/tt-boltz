import os, sys, time, json, random, traceback
sys.path.insert(0,"/home/moritz/tt-boltz2")
import torch; torch.set_grad_enabled(False)
import ttnn
from tt_boltz.tenstorrent import get_device, set_fast_mode
TAG=sys.argv[1]; REPO={"esmfold2-fast":"biohub/ESMFold2-Fast","esmfold2":"biohub/ESMFold2"}[TAG]
OUT=f"/tmp/bench_{TAG}.jsonl"
# ns=1 for every length (the working column, must complete incl 1024); ns=32 only
# where it can fit a single card (>=512 hard-crashes the process, so skip).
PLAN=[(L,1) for L in [32,64,128,256,512,1024]]+[(L,32) for L in [32,64,128,256]]
CFG=dict(num_loops=3, num_sampling_steps=14)
AA="ACDEFGHIKLMNPQRSTVWY"; def seq(L):
    r=random.Random(1234+L); return "".join(r.choice(AA) for _ in range(L))  # RNG once -> diverse seq
set_fast_mode(True); dev=get_device()
from tt_boltz.esmfold2_runtime import load_ttnn_esmfold2, fold_complex
m=load_ttnn_esmfold2(esmfold2_repo=REPO, fast=True); m._esmc.preload()
print(f"LOADED {TAG}", flush=True); open(OUT,"w").close()
for L,ns in PLAN:
    chains=[("A", seq(L), None)]
    try:
        fold_complex(m, chains, num_diffusion_samples=ns, seed=0, **CFG); ttnn.synchronize_device(dev)
        ts=[]
        for i in range(2):
            t=time.time(); fold_complex(m, chains, num_diffusion_samples=ns, seed=i+1, **CFG); ttnn.synchronize_device(dev); ts.append(time.time()-t)
        rec={"tag":TAG,"L":L,"ns":ns,"predict_s":round(min(ts),4),"runs":[round(x,4) for x in ts]}
    except Exception as e:
        rec={"tag":TAG,"L":L,"ns":ns,"predict_s":None,"error":f"{type(e).__name__}: {str(e)[:120]}"}; traceback.print_exc()
    open(OUT,"a").write(json.dumps(rec)+"\n")
    ps=rec.get("predict_s"); print(f"  {TAG} L={L} ns={ns} -> {ps} s ({round(1/ps,4) if ps else 'FAIL'} /s/card)", flush=True)
print(f"DONE {TAG}", flush=True)
