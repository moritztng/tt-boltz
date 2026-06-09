# On-device end-to-end sampler validation (deterministic, no RNG matching needed):
# replay the reference diffusion trajectory step-by-step using the PRODUCTION
# tt_bio.protenix.DiffusionModule. For each of the N_step denoiser calls, run
# DiffusionModule.denoise(x_noisy_i, t_hat_i, cond) with the fixed trunk conditioning
# (from protenix_denoiser_pre.pkl) and compare to the reference denoised coords
# (protenix_traj.pkl). Validates the productionized denoiser across the full sigma range.
import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2')
import pickle, torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import DiffusionModule

ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
D=pickle.load(open('/home/ttuser/protenix_denoiser_pre.pkl','rb')); kw=D['kwargs']
TRAJ=pickle.load(open('/home/ttuser/protenix_traj.pkl','rb')); STEPS=TRAJ['steps']
feat=kw['input_feature_dict']
s_inputs=kw['s_inputs'].float(); s_trunk=kw['s_trunk'].float(); pair_z=kw['pair_z'].float()
p_lm=kw['p_lm'].float()[0]; c_l=kw['c_l'].float()
N=c_l.shape[0]; NT=s_inputs.shape[0]; a2t=feat['atom_to_token_idx'].long(); mt=feat['pad_info']['mask_trunked'].float()
S=torch.zeros(N,NT); S[torch.arange(N),a2t]=1.0
cond={'s_trunk':s_trunk,'s_inputs':s_inputs,'pair_z':pair_z,'c_l':c_l,'p_lm':p_lm,'S':S,'mask_trunked':mt}

def pcc(u,v):
    u=u.flatten().double();v=v.flatten().double()
    return float(((u-u.mean())*(v-v.mean())).sum()/((u-u.mean()).norm()*(v-v.mean()).norm()))

dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
PRE='module.diffusion_module.'
dm_sd={k[len(PRE):]:v for k,v in ck.items() if k.startswith(PRE)}
dm=DiffusionModule(dm_sd, dev, ckc)

pccs=[]
for i,st in enumerate(STEPS):
    xn=st['x_noisy'].float(); th=st['t_hat'].float(); ref=st['denoised'].float()
    out=dm.denoise(xn, th, cond)
    p=pcc(out, ref[:, :N])
    pccs.append(p)
    print('step %2d  t_hat=%9.4g  denoised PCC %.5f  maxerr %.3e'%(i, float(th.max()), p, (out-ref[:,:N]).abs().max()), flush=True)
print('\nALL-STEP denoiser PCC: min %.5f  mean %.5f  (across t_hat %.3g..%.3g)'%(
    min(pccs), sum(pccs)/len(pccs), float(STEPS[-1]['t_hat'].max()), float(STEPS[0]['t_hat'].max())), flush=True)
