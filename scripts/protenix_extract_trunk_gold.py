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
orig=m.get_pairformer_output
def wrap(*a,**k):
    r=orig(*a,**k); cap['ret']=tuple(x.detach().clone() for x in r); return r
m.get_pairformer_output=wrap
dl=get_inference_dataloader(configs=cfg)
for b in dl:
    data,_,_=b[0]; break
m(input_feature_dict=data['input_feature_dict'], label_full_dict=None, label_dict=None, mode='inference', mc_dropout_apply_rate=0.0)
si,s,z=cap['ret']
d=pickle.load(open('/home/ttuser/protenix_ref_out.pkl','rb'))
dk=d['intermediates']['diffusion_module']['kwargs']
def pcc(a,b):
    a=a.flatten().double();b=b.flatten().double()
    return float(((a-a.mean())*(b-b.mean())).sum()/((a-a.mean()).norm()*(b-b.mean()).norm()))
print('returned trunk s',tuple(s.shape),'z',tuple(z.shape),flush=True)
print('pcc(returned_s, s_trunk)=%.5f'%pcc(s,dk['s_trunk'].float()),flush=True)
print('pcc(returned_z, pair_z)=%.5f'%pcc(z,dk['pair_z'].float()),flush=True)
print('returned_z absmean %.3e  pair_z absmean %.3e'%(z.abs().mean(),dk['pair_z'].abs().mean()),flush=True)
pickle.dump({'s':s,'z':z,'s_inputs':si}, open('/home/ttuser/protenix_trunk_gold.pkl','wb'))
print('saved trunk gold (returned s,z)', flush=True)
