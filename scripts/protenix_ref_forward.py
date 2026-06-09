# End-to-end Protenix-v2 reference forward (CPU).
# Builds the full 464.4M v2 model (CUDA FusedLayerNorm stubbed -> torch),
# loads real v2 weights, builds offline feats (no MSA/template/esm) for a tiny
# protein, forces torch triangle kernels (no cuequivariance/deepspeed CUDA), and
# runs the forward -> coords/plddt/pae saved to protenix_ref_out.pkl.
# Needs the py3.11 reference venv + PROTENIX_SRC repo checkout + CCD in ~/common/.
import sys, types, numbers, os, pickle
os.environ.setdefault('PROTENIX_ROOT_DIR', '/home/ttuser')
import os as _os; sys.path.insert(0, _os.environ.get('PROTENIX_SRC','/tmp/protenix-src'))
import torch, torch.nn as nn; from torch.nn import Parameter; import torch.nn.functional as F
torch.set_grad_enabled(False)

# --- stub CUDA FusedLayerNorm with a torch equivalent ---
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
from protenix.data.inference.infer_dataloader import get_inference_dataloader
from protenix.utils.torch_utils import to_device

# --- compose v2 config, offline (no MSA / template / esm), tiny diffusion ---
base = {**configs_base, **{"data": data_configs}, **inference_configs}
def du(d,u):
    for k,v in u.items():
        d[k]=du(d.get(k,{}),v) if isinstance(v,dict) and isinstance(d.get(k),dict) else v
    return d
du(base, model_configs['protenix-v2'])
base['input_json_path'] = '/home/ttuser/protenix_tiny.json'
base['dump_dir'] = '/home/ttuser/protenix_out'
base['use_msa'] = False
base['use_template'] = False
base['use_seeds_in_json'] = True
# force CPU-friendly torch kernels (no cuequivariance/deepspeed CUDA exts)
base['triangle_multiplicative'] = 'torch'
base['triangle_attention'] = 'torch'
if isinstance(base.get('esm'), dict): base['esm']['enable'] = False
# fast smoke: few diffusion steps / samples
base['sample_diffusion']['N_step'] = 10
base['sample_diffusion']['N_sample'] = 1
cfg = parse_configs(base, fill_required_with_null=True)

print('building v2 model...', flush=True)
m = Protenix(cfg).eval()
ck = torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt', map_location='cpu', weights_only=True)
ck = ck.get('model', ck)
sd = {k[len('module.'):] if k.startswith('module.') else k: v for k,v in ck.items()}
miss, unexp = m.load_state_dict(sd, strict=False)
print('V2 load: missing=%d unexpected=%d'%(len(miss), len(unexp)), flush=True)

print('building feats...', flush=True)
dl = get_inference_dataloader(configs=cfg)
batch = None
for b in dl:
    batch = b; break
data, atom_array, err = batch[0]
if err: print('DATA ERROR:', err, flush=True)
feat = data['input_feature_dict']
print('N_token=%d N_atom=%d N_msa=%d'%(int(data['N_token']), int(data['N_atom']), int(data['N_msa'])), flush=True)
print('feat keys (%d):'%len(feat), sorted(list(feat.keys()))[:30], flush=True)

cap = {}
if os.environ.get('DUMP_INTERMEDIATES'):
    def _cpu(x):
        import torch as _t
        if _t.is_tensor(x): return x.detach().cpu()
        if isinstance(x,(list,tuple)): return type(x)(_cpu(y) for y in x)
        if isinstance(x,dict): return {k:_cpu(v) for k,v in x.items()}
        return x
    def _mk(name):
        def hook(mod, args, kwargs, output):
            if name not in cap:  # first call only (cycle 0 / first diffusion step)
                cap[name] = {'in': _cpu(args), 'kwargs': _cpu(kwargs), 'out': _cpu(output)}
        return hook
    targets = {'input_embedder': m.input_embedder, 'msa_module': m.msa_module,
               'pairformer_stack': m.pairformer_stack, 'diffusion_module': m.diffusion_module,
               'confidence_head': m.confidence_head, 'template_embedder': m.template_embedder}
    for nm, mod in targets.items():
        mod.register_forward_hook(_mk(nm), with_kwargs=True)
    print('intermediate capture armed:', list(targets), flush=True)

print('running reference forward...', flush=True)
pred, _, _ = m(input_feature_dict=feat, label_full_dict=None, label_dict=None,
               mode='inference', mc_dropout_apply_rate=0.0)
for k,v in pred.items():
    if torch.is_tensor(v): print('  pred[%s]: %s'%(k, tuple(v.shape)), flush=True)
# save coords + feats for tt-bio comparison
coord_key = 'coordinate' if 'coordinate' in pred else ('coords' if 'coords' in pred else None)
out = {'pred': {k:(v.cpu() if torch.is_tensor(v) else v) for k,v in pred.items()},
       'feat': {k:(v.cpu() if torch.is_tensor(v) else v) for k,v in feat.items()}}
if cap:
    out['intermediates'] = cap
    print('captured intermediates:', list(cap), flush=True)
with open('/home/ttuser/protenix_ref_out.pkl','wb') as f:
    pickle.dump(out, f)
print('REF_FORWARD_DONE saved /home/ttuser/protenix_ref_out.pkl', flush=True)
