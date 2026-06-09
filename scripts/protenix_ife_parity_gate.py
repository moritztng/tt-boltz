# Parity gate for the Protenix-v2 InputFeatureEmbedder (atom encoder, has_coords=False).
# Loads input_embedder.* from the real v2 ckpt, feeds the golden feat captured in
# ~/protenix_ref_out.pkl, and confirms it reproduces the golden s_inputs (38,449).
# This is the standalone reference gate the tt-bio AtomAttentionEncoder validates vs.
# Run with the py3.11 reference venv. Result: PCC 1.0, maxabs_err 0.
import sys, types, numbers, os, pickle
os.environ.setdefault('PROTENIX_ROOT_DIR','/home/ttuser'); sys.path.insert(0,'/tmp/protenix-src')
import torch, torch.nn as nn; from torch.nn import Parameter; import torch.nn.functional as F
torch.set_grad_enabled(False)
stub = types.ModuleType('protenix.model.layer_norm.layer_norm')
class FusedLayerNorm(nn.Module):
    def __init__(s, ns, create_scale=True, create_offset=True, eps=1e-5, **kw):
        super().__init__()
        if isinstance(ns, numbers.Integral): ns=(ns,)
        s.normalized_shape=tuple(ns); s.eps=eps
        s.weight=Parameter(torch.ones(*ns)) if create_scale else None
        s.bias=Parameter(torch.zeros(*ns)) if create_offset else None
    def forward(s,x):
        x=F.layer_norm(x,s.normalized_shape,None,None,s.eps)
        if s.weight is not None: x=x*s.weight
        if s.bias is not None: x=x+s.bias
        return x
stub.FusedLayerNorm=FusedLayerNorm; sys.modules['protenix.model.layer_norm.layer_norm']=stub
from configs.configs_base import configs as cb
from configs.configs_data import data_configs as dc
from configs.configs_inference import inference_configs as ic
from configs.configs_model_type import model_configs as mc
from protenix.config.config import parse_configs
from protenix.model.modules.embedders import InputFeatureEmbedder
base={**cb,**{"data":dc},**ic}
def du(d,u):
    for k,v in u.items(): d[k]=du(d.get(k,{}),v) if isinstance(v,dict) and isinstance(d.get(k),dict) else v
    return d
du(base, mc['protenix-v2']); cfg=parse_configs(base, fill_required_with_null=True)
ife=InputFeatureEmbedder(**cfg.model.input_embedder, esm_configs=cfg.get('esm',{})).eval()
# load weights for input_embedder.* from the v2 ckpt
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
pre='input_embedder.'
sd={k[len('module.'+pre):]:v for k,v in ck.items() if k.startswith('module.'+pre)}
if not sd: sd={k[len(pre):]:v for k,v in ck.items() if k.startswith(pre)}
miss,unexp=ife.load_state_dict(sd,strict=False)
print('IFE load: missing=%d unexpected=%d'%(len(miss),len(unexp)),flush=True)
d=pickle.load(open('/home/ttuser/protenix_ref_out.pkl','rb'))
feat=d['intermediates']['input_embedder']['in'][0]  # the feat dict
gold=d['intermediates']['input_embedder']['out']
print('feat has d_lm/v_lm/pad_info:', all(k in feat for k in ['d_lm','v_lm','pad_info']), flush=True)
out=ife(feat)
err=(out-gold).abs().max().item()
def pcc(a,b):
    a=a.flatten().double(); b=b.flatten().double()
    return float(((a-a.mean())*(b-b.mean())).sum()/((a-a.mean()).norm()*(b-b.mean()).norm()))
print('s_inputs shape', tuple(out.shape), 'maxabs_err %.3e'%err, 'PCC %.6f'%pcc(out,gold), flush=True)
