# Validate the production tt_bio.protenix.Trunk class reproduces the full 10-cycle
# trunk vs the real v2 reference (golden s/z). Uses golden trunk inputs (s_inputs, relp,
# token_bonds, template/msa feats) from the captured intermediates.
import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2')
import pickle, torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import Trunk

ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
sd={k[len('module.'):] if k.startswith('module.') else k: v for k,v in ck.items()}
d=pickle.load(open('/home/ttuser/protenix_ref_out.pkl','rb'))
feat=d['intermediates']['template_embedder']['in'][0]
s_inputs=d['intermediates']['input_embedder']['out'].float()
tg=pickle.load(open('/home/ttuser/protenix_trunkin_gold.pkl','rb')); relp=tg['relp']; token_bonds=tg['token_bonds']
tgld=pickle.load(open('/home/ttuser/protenix_trunk_gold.pkl','rb')); s_gold=tgld['s'].float(); z_gold=tgld['z'].float()

dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
trunk=Trunk(sd, ckc)
def prog(stage,step,total): print('  %s cycle %d/%d'%(stage,step,total),flush=True)
s,z3=trunk(feat, s_inputs, relp, token_bonds, progress_fn=prog)
so=torch.Tensor(ttnn.to_torch(s)).float().reshape(s_gold.shape)
zo=torch.Tensor(ttnn.to_torch(z3)).float().reshape(z_gold.shape)
def pcc(a,b):
    a=a.flatten().double(); b=b.flatten().double()
    return float(((a-a.mean())*(b-b.mean())).sum()/((a-a.mean()).norm()*(b-b.mean()).norm()))
print('TRUNK CLASS (10 cycles)  s PCC %.5f  z PCC %.5f'%(pcc(so,s_gold),pcc(zo,z_gold)),flush=True)
