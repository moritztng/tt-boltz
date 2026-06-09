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
from protenix.model.protenix import Protenix
from protenix.data.inference.infer_dataloader import get_inference_dataloader
base={**cb,**{"data":dc},**ic}
def du(d,u):
    for k,v in u.items(): d[k]=du(d.get(k,{}),v) if isinstance(v,dict) and isinstance(d.get(k),dict) else v
    return d
du(base,mc['protenix-v2'])
base['input_json_path']='/home/ttuser/protenix_tiny.json'; base['dump_dir']='/home/ttuser/protenix_out'
base['use_msa']=False; base['use_template']=False; base['use_seeds_in_json']=True
base['triangle_multiplicative']='torch'; base['triangle_attention']='torch'
if isinstance(base.get('esm'),dict): base['esm']['enable']=False
base['sample_diffusion']['N_step']=10; base['sample_diffusion']['N_sample']=1
cfg=parse_configs(base,fill_required_with_null=True)
m=Protenix(cfg).eval()
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
m.load_state_dict({k[len('module.'):] if k.startswith('module.') else k:v for k,v in ck.items()},strict=False)
cap={}
dt=m.diffusion_module.diffusion_transformer
def _din(mod,a,kw):
    if 'din' not in cap: cap['din']=(kw.get('a',a[0] if a else None).detach().clone(), kw['s'].detach().clone(), kw['z'].detach().clone())
dt.register_forward_pre_hook(_din, with_kwargs=True)
blocks=dt.blocks
def mk(i):
    def h(mod,a,o):
        # block returns (a,s,z) tuple
        if i not in cap: cap[i]=(o[0].detach().clone() if isinstance(o,tuple) else o.detach().clone())
    return h
for i,b in enumerate(blocks): b.register_forward_hook(mk(i))
dl=get_inference_dataloader(configs=cfg)
for b in dl:
    data,_,_=b[0]; break
m(input_feature_dict=data['input_feature_dict'], label_full_dict=None, label_dict=None, mode='inference', mc_dropout_apply_rate=0.0)
pickle.dump(cap, open('/home/ttuser/protenix_dit_consistent.pkl','wb'))
print('consistent capture: din+', len([k for k in cap if isinstance(k,int)]),'blocks; a absmean', float(cap['din'][0].abs().mean()), flush=True)
