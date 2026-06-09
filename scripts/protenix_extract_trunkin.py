# Extract golden trunk-input (s_init,z_init) + LinearNoBias weights -> 
# ~/protenix_trunkin_gold.pkl (py3.11 venv). Gate for tt_bio.protenix.TrunkInput.
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
base={**cb,**{"data":dc},**ic}
def du(d,u):
    for k,v in u.items(): d[k]=du(d.get(k,{}),v) if isinstance(v,dict) and isinstance(d.get(k),dict) else v
    return d
du(base,mc['protenix-v2']); base['triangle_multiplicative']='torch'; base['triangle_attention']='torch'
cfg=parse_configs(base,fill_required_with_null=True)
m=Protenix(cfg).eval()
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
sd={k[len('module.'):] if k.startswith('module.') else k:v for k,v in ck.items()}
m.load_state_dict(sd,strict=False)
d=pickle.load(open('/home/ttuser/protenix_ref_out.pkl','rb'))
feat=d['intermediates']['input_embedder']['in'][0]
s_inputs=d['intermediates']['input_embedder']['out']  # golden (38,449)
# relp must be generated (generate_relp) - it's already in feat (captured post update). check:
relp = feat['relp'] if 'relp' in feat else m.relative_position_encoding.generate_relp(feat)['relp']
s_init=m.linear_no_bias_sinit(s_inputs)
z_init=m.linear_no_bias_zinit1(s_init)[...,None,:]+m.linear_no_bias_zinit2(s_init)[...,None,:,:]
z_init=z_init+m.relative_position_encoding(relp)
z_init=z_init+m.linear_no_bias_token_bond(feat['token_bonds'].unsqueeze(-1))
zc=m.constraint_embedder(feat['constraint_feature']) if 'constraint_feature' in feat else None
if zc is not None: z_init=z_init+zc
print('s_init',tuple(s_init.shape),'z_init',tuple(z_init.shape),'zc',None if zc is None else tuple(zc.shape),flush=True)
# collect trunk-input weights
keys=['linear_no_bias_sinit','linear_no_bias_zinit1','linear_no_bias_zinit2','linear_no_bias_token_bond','relative_position_encoding','constraint_embedder']
wsd={k:v.detach().clone() for k,v in m.state_dict().items() if any(k.startswith(p+'.') or k==p+'.weight' for p in keys)}
out={'s_inputs':s_inputs,'relp':relp,'token_bonds':feat['token_bonds'],'constraint_feature':feat['constraint_feature'],
     'golden_s_init':s_init,'golden_z_init':z_init,'weights':wsd}
pickle.dump(out, open('/home/ttuser/protenix_trunkin_gold.pkl','wb'))
print('trunk-in weight keys:', sorted(wsd.keys()), flush=True); print('EXTRACT_DONE',flush=True)
