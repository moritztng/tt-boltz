import os,sys
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2')
import pickle, torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import DiffusionModule
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
D=pickle.load(open('/home/ttuser/protenix_denoiser_pre.pkl','rb')); kw=D['kwargs']
T=pickle.load(open('/home/ttuser/protenix_traj.pkl','rb')); S=T['steps']; sch=T['sched']; fin=T['final_coords'].float().reshape(-1,3)
feat=kw['input_feature_dict']
s_inputs=kw['s_inputs'].float(); s_trunk=kw['s_trunk'].float(); pair_z=kw['pair_z'].float()
p_lm=kw['p_lm'].float()[0]; c_l=kw['c_l'].float()
N=c_l.shape[0]; NT=s_inputs.shape[0]; a2t=feat['atom_to_token_idx'].long(); mt=feat['pad_info']['mask_trunked'].float()
S_mat=torch.zeros(N,NT); S_mat[torch.arange(N),a2t]=1.0
cond={'s_trunk':s_trunk,'s_inputs':s_inputs,'pair_z':pair_z,'c_l':c_l,'p_lm':p_lm,'S':S_mat,'mask_trunked':mt}
dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
dm=DiffusionModule({k[len('module.diffusion_module.'):]:v for k,v in ck.items() if k.startswith('module.diffusion_module.')}, dev, ckc)
def kab(P,Q):
    P=P-P.mean(0);Q=Q-Q.mean(0);H=P.t()@Q;U,_,Vt=torch.linalg.svd(H)
    d=torch.sign(torch.det(Vt.t()@U.t()));Dm=torch.diag(torch.tensor([1.,1.,d]));R=Vt.t()@Dm@U.t()
    return float(((P@R.t())-Q).pow(2).sum(-1).mean().sqrt())
ss=sch['step_scale_eta']; ns=len(S)
# my denoiser on the reference's last-step noise -> EDM final update -> vs reference final
k=ns-1
xn=S[k]['x_noisy'].float(); th=S[k]['t_hat'].float()
my_dn=dm.denoise(xn, th, cond)[0][:N]           # my on-device denoised at last step
ref_dn=S[k]['denoised'].float().reshape(-1,3)[:N]
xn2=xn.reshape(-1,3)[:N]
my_final = xn2 + ss*(0.0 - float(th))* (xn2 - my_dn)/float(th)
print('MY denoiser vs REF denoised (last step) Kabsch: %.3f A'%kab(my_dn, ref_dn))
print('MY deterministic final  vs REF final_coords Kabsch: %.3f A'%kab(my_final, fin), flush=True)
