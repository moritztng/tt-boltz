import sys, types, numbers, os, pickle
os.environ.setdefault('PROTENIX_ROOT_DIR','/home/ttuser'); sys.path.insert(0,'/tmp/protenix-src')
import torch, torch.nn as nn; from torch.nn import Parameter; import torch.nn.functional as F
torch.set_grad_enabled(False)
stub=types.ModuleType('protenix.model.layer_norm.layer_norm')
class FLN(nn.Module):
    def __init__(s,ns,create_scale=True,create_offset=True,eps=1e-5,**kw):
        super().__init__()
        if isinstance(ns,numbers.Integral): ns=(ns,)
        s.normalized_shape=tuple(ns); s.eps=eps
        s.weight=Parameter(torch.ones(*ns)) if create_scale else None
        s.bias=Parameter(torch.zeros(*ns)) if create_offset else None
    def forward(s,x):
        x=F.layer_norm(x,s.normalized_shape,None,None,s.eps)
        if s.weight is not None: x=x*s.weight
        if s.bias is not None: x=x+s.bias
        return x
stub.FusedLayerNorm=FLN; sys.modules['protenix.model.layer_norm.layer_norm']=stub
from configs.configs_base import configs as cb
from configs.configs_data import data_configs as dc
from configs.configs_inference import inference_configs as ic
from configs.configs_model_type import model_configs as mc
from protenix.config.config import parse_configs
from protenix.model.modules.diffusion import DiffusionModule
from protenix.model.generator import sample_diffusion, InferenceNoiseScheduler
base={**cb,**{"data":dc},**ic}
def du(d,u):
    for k,v in u.items(): d[k]=du(d.get(k,{}),v) if isinstance(v,dict) and isinstance(d.get(k),dict) else v
    return d
du(base,mc['protenix-v2']); base['triangle_multiplicative']='torch'; base['triangle_attention']='torch'
cfg=parse_configs(base,fill_required_with_null=True)
dm=DiffusionModule(**cfg.model.diffusion_module).eval()
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
dm.load_state_dict({k[len('module.diffusion_module.'):]:v for k,v in ck.items() if k.startswith('module.diffusion_module.')},strict=False)
D=pickle.load(open('/home/ttuser/protenix_denoiser_pre.pkl','rb'))['kwargs']
sched=InferenceNoiseScheduler()(N_step=20)
kw=dict(input_feature_dict=D['input_feature_dict'], s_inputs=D['s_inputs'], s_trunk=D['s_trunk'], z_trunk=D['z_trunk'], pair_z=D['pair_z'], p_lm=D['p_lm'], c_l=D['c_l'], noise_schedule=sched, N_sample=1, gamma0=0.8, gamma_min=1.0, noise_scale_lambda=1.003, step_scale_eta=1.5)
def run(seed):
    torch.manual_seed(seed)
    return sample_diffusion(denoise_net=dm, **kw)[0].float()  # (N_atom,3)
# fp32 reference
torch.manual_seed(0); c_fp32=run(0)
# bf16 DiT: wrap diffusion_transformer to cast a/s/z to bf16
_orig=dm.diffusion_transformer.forward
def bf16_dt(*a,**k):
    k=dict(k)
    for key in ('a','s','z'):
        if key in k and torch.is_tensor(k[key]): k[key]=k[key].to(torch.bfloat16).to(torch.float32)
    return _orig(*a,**k)
dm.diffusion_transformer.forward=bf16_dt
c_bf16=run(0)
def kabsch_rmsd(P,Q):
    P=P-P.mean(0); Q=Q-Q.mean(0)
    H=P.T@Q; U,S,Vt=torch.linalg.svd(H); d=torch.sign(torch.det(Vt.T@U.T)); D=torch.diag(torch.tensor([1.,1.,d])); R=Vt.T@D@U.T
    Pr=(R@P.T).T
    return float(torch.sqrt(((Pr-Q)**2).sum(-1).mean()))
print('Ca-RMSD fp32-DiT vs bf16-DiT sampler (same seed, all-atom) = %.3f A'%kabsch_rmsd(c_fp32,c_bf16), flush=True)
print('coord absmean fp32 %.2f bf16 %.2f'%(c_fp32.abs().mean(), c_bf16.abs().mean()), flush=True)
