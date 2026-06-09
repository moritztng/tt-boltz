import os, sys, re
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2'); sys.path.insert(0,'/home/ttuser/tt-boltz2/tests')
import pickle, torch, torch.nn.functional as F, ttnn
from protenix_reference import remap_pairformer_block
from tt_bio.tenstorrent import get_device, Pairformer, CORE_GRID_MAIN as CORE
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
P='module.confidence_head.'; g=lambda k: ck[P+k].float()
gc=pickle.load(open('/home/ttuser/protenix_confidence_pre.pkl','rb')); kw=gc['kwargs']
feat=kw['input_feature_dict']; s_inputs=kw['s_inputs'].float(); s_trunk=kw['s_trunk'].float(); z_trunk=kw['z_trunk'].float()
x_pred=kw['x_pred_coords'].float()
plddt_g,pae_g,pde_g,resolved_g=[t.float() for t in gc['out']]
N=s_inputs.shape[-2] if s_inputs.dim()>1 else s_inputs.shape[0]
s_inputs=s_inputs.reshape(-1,449)[:N] if s_inputs.dim()>2 else s_inputs.reshape(N,449)
s_trunk=s_trunk.reshape(N,384); z_trunk=z_trunk.reshape(N,N,256)
# host-side prep (distance embed) then pairformer on device
import math
s_t=F.layer_norm(torch.clamp(s_trunk,-512,512),(384,))*g('input_strunk_ln.weight')+ (g('input_strunk_ln.bias') if (P+'input_strunk_ln.bias') in ck else 0)
mask=feat['distogram_rep_atom_mask'].bool()
xr=x_pred.reshape(-1,3)[mask] if x_pred.dim()==2 else x_pred.reshape(x_pred.shape[-2],3)[mask]
z=z_trunk + F.linear(s_inputs,g('linear_no_bias_s1.weight'))[None,:,:].transpose(0,1) + F.linear(s_inputs,g('linear_no_bias_s2.weight'))[None,:,:]
# fix broadcast: s1 -> [N,1,c], s2 -> [1,N,c]
z=z_trunk + F.linear(s_inputs,g('linear_no_bias_s1.weight')).unsqueeze(1) + F.linear(s_inputs,g('linear_no_bias_s2.weight')).unsqueeze(0)
d=torch.cdist(xr,xr)
lb=g('lower_bins'); ub=g('upper_bins')
oh=((d.unsqueeze(-1)>=lb)&(d.unsqueeze(-1)<ub)).float()
z=z + F.linear(oh,g('linear_no_bias_d.weight')) + F.linear(d.unsqueeze(-1),g('linear_no_bias_d_wo_onehot.weight'))
# pairformer on device (4 blocks, conf weights)
dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
nb=1+max(int(re.search(r'pairformer_stack\.blocks\.(\d+)\.',k).group(1)) for k in ck if k.startswith(P+'pairformer_stack.blocks.'))
comb={}
for i in range(nb):
    bsd={k[len(P+f'pairformer_stack.blocks.{i}.'):]:v for k,v in ck.items() if k.startswith(P+f'pairformer_stack.blocks.{i}.')}
    for kk,vv in remap_pairformer_block(bsd).items(): comb[f'layers.{i}.{kk}']=vv
b0=P+'pairformer_stack.blocks.0.'; nhp=ck[b0+'tri_att_start.linear.weight'].shape[0]; chpa=ck[b0+'tri_att_start.mha.linear_q.weight'].shape[0]//nhp; apb_nh=ck[b0+'attention_pair_bias.linear_nobias_z.weight'].shape[0]
pf=Pairformer(nb,chpa,nhp,384//apb_nh,apb_nh,True,comb,ckc)
ft=lambda x: ttnn.from_torch(x.float(),layout=ttnn.TILE_LAYOUT,device=dev,dtype=ttnn.bfloat16)
so,zo=pf(ft(s_t.unsqueeze(0)),ft(z.unsqueeze(0)))
s_single=torch.Tensor(ttnn.to_torch(so)).float().reshape(N,384); zf=torch.Tensor(ttnn.to_torch(zo)).float().reshape(N,N,256)
# heads (host, small)
pae=F.linear(F.layer_norm(zf,(256,))*g('pae_ln.weight')+ (g('pae_ln.bias') if (P+'pae_ln.bias') in ck else 0), g('linear_no_bias_pae.weight'))
pde=F.linear(F.layer_norm(zf+zf.transpose(0,1),(256,))*g('pde_ln.weight')+ (g('pde_ln.bias') if (P+'pde_ln.bias') in ck else 0), g('linear_no_bias_pde.weight'))
a2t=feat['atom_to_token_idx'].long(); a2ta=feat['atom_to_tokatom_idx'].long()
a=s_single[a2t]
aln=F.layer_norm(a,(384,))*g('plddt_ln.weight')+ (g('plddt_ln.bias') if (P+'plddt_ln.bias') in ck else 0)
plddt=torch.einsum('nc,ncb->nb', aln, g('plddt_weight')[a2ta])
rln=F.layer_norm(a,(384,))*g('resolved_ln.weight')+ (g('resolved_ln.bias') if (P+'resolved_ln.bias') in ck else 0)
resolved=torch.einsum('nc,ncb->nb', rln, g('resolved_weight')[a2ta])
def pcc(u,v):
    u=u.flatten().double();v=v.flatten().double()
    return float(((u-u.mean())*(v-v.mean())).sum()/((u-u.mean()).norm()*(v-v.mean()).norm()))
print('pae %.4f pde %.4f plddt %.4f resolved %.4f'%(pcc(pae,pae_g),pcc(pde,pde_g),pcc(plddt,plddt_g),pcc(resolved,resolved_g)),flush=True)
