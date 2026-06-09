# Capture the FULL per-step diffusion trajectory of the v2 reference sampler:
# every diffusion_module (denoiser) call's (x_noisy, t_hat) inputs + denoised output,
# plus the sigma schedule and final coords. Lets the on-device denoiser be validated
# at EVERY sigma (not just step 0), and the EDM-update arithmetic checked deterministically.
# Reuses /tmp/cap_diff.py model+cfg+stub setup. Run in the py3.11 reference venv.
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
exec(open('/tmp/cap_diff.py').read().split("dm=m.diffusion_module")[0])  # build m + cfg

# capture EVERY denoiser call (pre-hook clones inputs before in-place mutation)
steps=[]
pend={}
def _pre(mod,args,kwargs):
    pend['x']=kwargs['x_noisy'].detach().clone()
    pend['t']=kwargs['t_hat_noise_level'].detach().clone()
def _post(mod,a,o):
    out=o.detach().clone() if torch.is_tensor(o) else o[0].detach().clone()
    steps.append({'x_noisy':pend['x'],'t_hat':pend['t'],'denoised':out})
m.diffusion_module.register_forward_pre_hook(_pre, with_kwargs=True)
m.diffusion_module.register_forward_hook(_post)

from protenix.data.inference.infer_dataloader import get_inference_dataloader
dl=get_inference_dataloader(configs=cfg)
for b in dl:
    data,_,_=b[0]; break
pred,_,_=m(input_feature_dict=data['input_feature_dict'], label_full_dict=None, label_dict=None,
           mode='inference', mc_dropout_apply_rate=0.0)

# sigma schedule from the inference noise scheduler (mirror config)
sd=cfg.sample_diffusion
sched={'N_step':int(sd['N_step']),'N_sample':int(sd['N_sample']),
       'gamma0':float(sd.get('gamma0',0.8)),'gamma_min':float(sd.get('gamma_min',1.0)),
       'noise_scale_lambda':float(sd.get('noise_scale_lambda',1.003)),
       'step_scale_eta':float(sd.get('step_scale_eta',1.5)),
       'rho':float(sd.get('rho',7)),'sigma_data':16.0}
out={'steps':steps,'sched':sched,
     'final_coords':pred['coordinate'].detach().cpu().float() if 'coordinate' in pred else None}
pickle.dump(out, open('/home/ttuser/protenix_traj.pkl','wb'))
print('TRAJ: %d denoiser calls; t_hat range [%.4g, %.4g]; sched=%s'%(
    len(steps), float(steps[-1]['t_hat'].max()), float(steps[0]['t_hat'].max()), sched), flush=True)
