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
exec(open('/tmp/cap_diff.py').read().split("dm=m.diffusion_module")[0])  # build m + feat (reuse setup)
# pre-hook to capture CLONED inputs BEFORE mutation
cap={}
def _pre(mod,args,kwargs):
    if 'in' not in cap:
        cap['in']=tuple(a.detach().clone() if torch.is_tensor(a) else (a.copy() if isinstance(a,dict) else a) for a in args)
        cap['kwargs']={k:(v.detach().clone() if torch.is_tensor(v) else v) for k,v in kwargs.items()}
def _post(mod,a,o):
    if 'out' not in cap: cap['out']=tuple(x.detach().clone() if torch.is_tensor(x) else x for x in o)
aae=m.diffusion_module.atom_attention_encoder
aae.register_forward_pre_hook(_pre, with_kwargs=True)
aae.register_forward_hook(_post)
from protenix.data.inference.infer_dataloader import get_inference_dataloader
dl=get_inference_dataloader(configs=cfg)
for b in dl:
    data,_,_=b[0]; break
m(input_feature_dict=data['input_feature_dict'], label_full_dict=None, label_dict=None, mode='inference', mc_dropout_apply_rate=0.0)
pickle.dump(cap, open('/home/ttuser/protenix_atomenc_pre.pkl','wb'))
print('pre-hook captured: in',len(cap['in']),'kwargs',list(cap['kwargs']),'out',len(cap['out']),flush=True)
