# Full diffusion-sampler end-to-end check: run tt_bio.protenix.edm_sample with the
# production DiffusionModule + golden trunk conditioning (protenix_denoiser_pre.pkl),
# N_step=10 (matching the reference run), and report sanity + Kabsch-aligned RMSD vs the
# reference final coords (protenix_traj.pkl). Since RNG (augmentation/noise) differs from
# the reference, this yields a different-but-valid sample; RMSD reflects sample variance.
import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2')
import pickle, torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import DiffusionModule, edm_sample

ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
D=pickle.load(open('/home/ttuser/protenix_denoiser_pre.pkl','rb')); kw=D['kwargs']
TRAJ=pickle.load(open('/home/ttuser/protenix_traj.pkl','rb'))
feat=kw['input_feature_dict']
s_inputs=kw['s_inputs'].float(); s_trunk=kw['s_trunk'].float(); pair_z=kw['pair_z'].float()
p_lm=kw['p_lm'].float()[0]; c_l=kw['c_l'].float()
N=c_l.shape[0]; NT=s_inputs.shape[0]; a2t=feat['atom_to_token_idx'].long(); mt=feat['pad_info']['mask_trunked'].float()
S=torch.zeros(N,NT); S[torch.arange(N),a2t]=1.0
cond={'s_trunk':s_trunk,'s_inputs':s_inputs,'pair_z':pair_z,'c_l':c_l,'p_lm':p_lm,'S':S,'mask_trunked':mt}

dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
PRE='module.diffusion_module.'
dm=DiffusionModule({k[len(PRE):]:v for k,v in ck.items() if k.startswith(PRE)}, dev, ckc)

def kabsch_rmsd(P,Q):
    Pc=P-P.mean(0); Qc=Q-Q.mean(0)
    H=Pc.t()@Qc; U,_,Vt=torch.linalg.svd(H)
    dsign=torch.sign(torch.det(Vt.t()@U.t()))
    Dm=torch.diag(torch.tensor([1.,1.,dsign]))
    R=(Vt.t()@Dm@U.t()); Pa=Pc@R.t()
    return float((Pa-Qc).pow(2).sum(-1).mean().sqrt())

x0=edm_sample(dm, cond, N, n_step=10, seed=0)[0]   # (N,3)
x1=edm_sample(dm, cond, N, n_step=10, seed=1)[0]
print('sampled coords shape %s  finite=%s'%(tuple(x0.shape), bool(torch.isfinite(x0).all())), flush=True)
print('radius of gyration %.3f A   pairwise-dist mean %.3f A'%(
    float((x0-x0.mean(0)).pow(2).sum(-1).mean().sqrt()), float(torch.pdist(x0).mean())), flush=True)
print('seed-to-seed Kabsch RMSD (my sampler, two seeds): %.3f A'%kabsch_rmsd(x0,x1), flush=True)
ref=TRAJ.get('final_coords')
if ref is not None:
    rf=ref.float().reshape(-1,3)[:N]
    print('Kabsch RMSD seed0 vs reference final coords: %.3f A (N=%d)'%(kabsch_rmsd(x0,rf),N), flush=True)
    print('Kabsch RMSD seed1 vs reference final coords: %.3f A'%kabsch_rmsd(x1,rf), flush=True)
print('SAMPLE_E2E_DONE', flush=True)
