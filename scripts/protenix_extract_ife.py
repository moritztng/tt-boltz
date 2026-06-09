# Extract the full AtomAttentionEncoder (has_coords=False) weights + feat fields
# + golden s_inputs -> ~/protenix_ife_gold.pkl (py3.11 venv). Gate for the on-device
# tt_bio.protenix.AtomAttentionEncoder (tests/test_protenix_ife.py).
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
d=pickle.load(open('/home/ttuser/protenix_ref_out.pkl','rb'))
feat=d['intermediates']['input_embedder']['in'][0]
gold_sinputs=d['intermediates']['input_embedder']['out']
out={'aae_state': {k:v.detach().clone() for k,v in ife.atom_attention_encoder.state_dict().items()},
     'feat':{k:feat[k] for k in ['ref_pos','ref_charge','ref_mask','ref_element','ref_atom_name_chars','d_lm','v_lm','atom_to_token_idx','restype','profile','deletion_mean']},
     'mask_trunked':feat['pad_info']['mask_trunked'],
     'golden_sinputs':gold_sinputs}
pickle.dump(out, open('/home/ttuser/protenix_ife_gold.pkl','wb'))
print('aae params:', len(out['aae_state']), 'golden s_inputs', tuple(gold_sinputs.shape), flush=True)
print('small_mlp/cl/cm/q keys:', [k for k in out['aae_state'] if any(t in k for t in ['small_mlp','linear_no_bias_cl','linear_no_bias_cm','linear_no_bias_q'])], flush=True)
print('EXTRACT_DONE', flush=True)
