# Extract golden atom-transformer I/O (q,c,p -> qout) + the 3-block weights
# from the real v2 reference -> ~/protenix_atomtx_gold.pkl (py3.11 venv).
# Validation target for the ttnn local windowed atom attention.
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
from protenix.model.modules.embedders import InputFeatureEmbedder
base={**cb,**{"data":dc},**ic}
def du(d,u):
    for k,v in u.items(): d[k]=du(d.get(k,{}),v) if isinstance(v,dict) and isinstance(d.get(k),dict) else v
    return d
du(base,mc['protenix-v2']); cfg=parse_configs(base,fill_required_with_null=True)
ife=InputFeatureEmbedder(**cfg.model.input_embedder,esm_configs=cfg.get('esm',{})).eval()
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
pre='input_embedder.'; sd={k[len('module.'+pre):]:v for k,v in ck.items() if k.startswith('module.'+pre)}
ife.load_state_dict(sd,strict=False)
aae=ife.atom_attention_encoder
d=pickle.load(open('/home/ttuser/protenix_ref_out.pkl','rb'))
feat=d['intermediates']['input_embedder']['in'][0]
cap={}
def hook(m,a,kw,o):
    cap['in']=(a, {k:(v.detach().clone() if torch.is_tensor(v) else v) for k,v in kw.items()}); cap['out']=o.detach().clone()
aae.atom_transformer.register_forward_hook(hook, with_kwargs=True)
a_out=ife(feat)
qin, kw = cap['in']
print('atom_tx pos-args:', [tuple(x.shape) for x in qin], 'kwargs:', list(kw.keys()), flush=True)
print('atom_tx out:', tuple(cap['out'].shape), flush=True)
# atom_transformer.forward(q, c, p) -> positional q,c,p
out={'q':qin[0].detach().clone(),'c':qin[1].detach().clone(),'p':qin[2].detach().clone(),
     'mask_trunked':feat['pad_info']['mask_trunked'].detach().clone(),
     'golden_qout':cap['out'],
     'weights':{n:p.detach().clone() for n,p in aae.atom_transformer.named_parameters()}}
pickle.dump(out, open('/home/ttuser/protenix_atomtx_gold.pkl','wb'))
print('n weights:', len(out['weights']), flush=True)
print('weight names sample:', list(out['weights'].keys())[:12], flush=True)
print('EXTRACT_DONE', flush=True)
