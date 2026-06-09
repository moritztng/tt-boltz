# Builds the full Protenix-v2 reference model + loads real v2 weights.
# Requires: a Protenix repo checkout (PROTENIX_SRC, for configs/ + runner/),
# protenix pip pkg + deps (absl-py, biotite==1.0.1, fair-esm, modelcif), and the
# v2 checkpoint. Stubs the CUDA FusedLayerNorm with a torch equivalent.
import sys, types, numbers; import os; sys.path.insert(0, os.environ.get('PROTENIX_SRC','/tmp/protenix-src'))
import torch, torch.nn as nn; from torch.nn import Parameter; import torch.nn.functional as F
torch.set_grad_enabled(False)
stub = types.ModuleType('protenix.model.layer_norm.layer_norm')
class FusedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, create_scale=True, create_offset=True, eps=1e-5, **kw):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral): normalized_shape=(normalized_shape,)
        self.normalized_shape=tuple(normalized_shape); self.eps=eps
        self.weight = Parameter(torch.ones(*normalized_shape)) if create_scale else None
        self.bias = Parameter(torch.zeros(*normalized_shape)) if create_offset else None
    def forward(self, x):
        x = F.layer_norm(x, self.normalized_shape, None, None, self.eps)
        if self.weight is not None: x = x*self.weight
        if self.bias is not None: x = x+self.bias
        return x
stub.FusedLayerNorm = FusedLayerNorm
sys.modules['protenix.model.layer_norm.layer_norm'] = stub
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs
from protenix.model.protenix import Protenix
print('model_configs keys:', list(model_configs.keys()), flush=True)
base = {**configs_base, **{"data": data_configs}, **inference_configs}
def du(d,u):
    for k,v in u.items():
        d[k]=du(d.get(k,{}),v) if isinstance(v,dict) and isinstance(d.get(k),dict) else v
    return d
mn = 'protenix-v2'
du(base, model_configs[mn])
cfg = parse_configs(base, fill_required_with_null=True)
print('building v2 model...', flush=True)
m = Protenix(cfg).eval()
print('v2 model: %.1fM params'%(sum(p.numel() for p in m.parameters())/1e6), flush=True)
ck = torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt', map_location='cpu', weights_only=True)
ck = ck.get('model', ck)
sd = {k[len('module.'):] if k.startswith('module.') else k: v for k,v in ck.items()}
miss, unexp = m.load_state_dict(sd, strict=False)
print('V2 load: missing=%d unexpected=%d'%(len(miss), len(unexp)), flush=True)
if miss: print(' missing:', miss[:6], flush=True)
if unexp: print(' unexpected:', unexp[:6], flush=True)
print('REF_BUILD_DONE', flush=True)
