# Extract golden atom-featurization (c_l, p_lm) + inputs + real v2 weights
# from the v2 reference into ~/protenix_atomfeat_gold.pkl. Run with the py3.11
# reference venv. The on-device tt-bio test (tests/test_protenix_atomfeat.py)
# loads this pkl so it needs NO protenix install in system python3.
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
# call prepare_cache to get golden c_l, p_lm (has_coords=False -> r_l=None,z=None)
p_lm, c_l = aae.prepare_cache(ref_pos=feat['ref_pos'], ref_charge=feat['ref_charge'],
    ref_mask=feat['ref_mask'], ref_element=feat['ref_element'],
    ref_atom_name_chars=feat['ref_atom_name_chars'], atom_to_token_idx=feat['atom_to_token_idx'],
    d_lm=feat['d_lm'], v_lm=feat['v_lm'], pad_info=feat['pad_info'])
print('c_l', tuple(c_l.shape), 'p_lm', tuple(p_lm.shape), flush=True)
out={'inputs':{k:feat[k] for k in ['ref_pos','ref_charge','ref_mask','ref_element','ref_atom_name_chars','d_lm','v_lm']},
     'mask_trunked':feat['pad_info']['mask_trunked'],
     'weights':{n:p.detach().clone() for n,p in aae.named_parameters() if any(t in n for t in ['linear_no_bias_ref_pos','linear_no_bias_ref_charge','linear_no_bias_f','linear_no_bias_d','linear_no_bias_invd','linear_no_bias_v'])},
     'golden_c_l':c_l, 'golden_p_lm':p_lm}
pickle.dump(out, open('/home/ttuser/protenix_atomfeat_gold.pkl','wb'))
print('weights:', list(out['weights'].keys()), flush=True)
print('EXTRACT_DONE', flush=True)
