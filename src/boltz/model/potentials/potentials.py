from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Set, List, Union

import torch
import numpy as np
from boltz.data import const
from boltz.model.potentials.schedules import ParameterSchedule, ExponentialInterpolation, PiecewiseStepFunction
from boltz.model.loss.diffusionv2 import weighted_rigid_align

class Potential(ABC):
    def __init__(
        self,
        parameters: Optional[Dict[str, Union[ParameterSchedule, float, int, bool]]] = None
    ):
        self.parameters = parameters

    def compute(self, coords, feats, parameters):
        index, args, com_args, ref_args, operator_args = self.compute_args(feats, parameters)

        if index.shape[1] == 0:
            return torch.zeros(coords.shape[:-2], device=coords.device)

        ref_coords = self.get_reference_coords(feats, parameters)

        if operator_args is not None:
            negation_mask, union_index = operator_args
        else:
            negation_mask, union_index = None, None
        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_com_index = com_index[atom_pad_mask]
            unpad_coords = coords[...,atom_pad_mask,:]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                'mean'
            )   
        if ref_args is not None:
            atom_token_id, ref_atom_idx = ref_args
            coords = coords[:, ref_atom_idx]

        value = self.compute_variable(coords, index, ref_coords=ref_coords, compute_gradient=False)
        energy = self.compute_function(value, *args, negation_mask=negation_mask, compute_derivative=False)
        normalization = self.compute_normalization(feats, parameters)

        if union_index is not None:
            neg_exp_energy = torch.exp(-1 * parameters['union_lambda'] * energy)
            Z = torch.zeros((*energy.shape[:-1], union_index.max() + 1), device=union_index.device).scatter_reduce(
                -1,
                union_index.expand_as(neg_exp_energy),
                neg_exp_energy,
                'sum',
            )
            softmax_energy = neg_exp_energy / Z[...,union_index]
            softmax_energy[Z[...,union_index] == 0] = 0
            return (energy * softmax_energy).sum(dim=-1) / normalization


        return energy.sum(dim=-1) / normalization

    def compute_gradient(self, coords, feats, parameters):
        index, args, com_args, ref_args, operator_args = self.compute_args(feats, parameters)
        if index.shape[1] == 0:
            return torch.zeros_like(coords)

        ref_coords = self.get_reference_coords(feats, parameters)

        if operator_args is not None:
            negation_mask, union_index = operator_args
        else:
            negation_mask, union_index = None, None
        if com_args is not None:
            com_index, atom_pad_mask = com_args
        else:
            com_index, atom_pad_mask = None, None
        if ref_args is not None:
            atom_token_id, ref_atom_idx = ref_args
        else:
            atom_token_id, ref_atom_idx = None, None


        if com_index is not None:
            unpad_coords = coords[...,atom_pad_mask,:]
            unpad_com_index = com_index[atom_pad_mask]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                'mean'
            )
            com_counts = torch.bincount(com_index[atom_pad_mask])
        elif ref_atom_idx is not None:
            coords = coords[:, ref_atom_idx]

        value, grad_value = self.compute_variable(coords, index, ref_coords=ref_coords, compute_gradient=True)
        energy, dEnergy = self.compute_function(value, *args, negation_mask=negation_mask, compute_derivative=True)
        if union_index is not None:
            neg_exp_energy = torch.exp(-1 * parameters['union_lambda'] * energy)
            Z = torch.zeros((*energy.shape[:-1], union_index.max() + 1), device=union_index.device).scatter_reduce(
                -1,
                union_index.expand_as(energy),
                neg_exp_energy,
                'sum',
            )
            softmax_energy = neg_exp_energy / Z[...,union_index]
            softmax_energy[Z[...,union_index] == 0] = 0
            f = torch.zeros((*energy.shape[:-1], union_index.max() + 1), device=union_index.device).scatter_reduce(
                -1,
                union_index.expand_as(energy),
                energy * softmax_energy,
                'sum',
            )
            dSoftmax = dEnergy * softmax_energy * (1 + parameters['union_lambda'] * (energy - f[...,union_index]))
            grad_atom = torch.zeros_like(coords).scatter_reduce(
                -2,
                index.flatten(start_dim=0, end_dim=1).unsqueeze(-1).expand((*coords.shape[:-2],-1,3)),
                dSoftmax.tile(grad_value.shape[-3]).unsqueeze(-1) * grad_value.flatten(start_dim=-3, end_dim=-2),
                'sum',
            )
        else:
            grad_atom = torch.zeros_like(coords).scatter_reduce(
                -2,
                index.flatten(start_dim=0, end_dim=1).unsqueeze(-1).expand((*coords.shape[:-2],-1,3)),
                dEnergy.tile(grad_value.shape[-3]).unsqueeze(-1) * grad_value.flatten(start_dim=-3, end_dim=-2),
                'sum',
            )

        if com_index is not None:
            grad_atom = grad_atom[..., com_index, :]
        elif atom_token_id is not None:
            grad_atom = grad_atom[..., atom_token_id, :]
        
        return grad_atom

    def compute_parameters(self, t):
        if self.parameters is None:
            return None
        parameters = {
            name: parameter if not isinstance(parameter, ParameterSchedule) else parameter.compute(t)
            for name, parameter in self.parameters.items()
        }
        return parameters

    @abstractmethod
    def compute_function(self, value, *args, negation_mask=None, compute_derivative=False):
        raise NotImplementedError

    @abstractmethod
    def compute_variable(self, coords, index, compute_gradient=False):
        raise NotImplementedError

    @abstractmethod
    def compute_args(self, t, feats, **parameters):
        raise NotImplementedError

    @abstractmethod
    def compute_normalization(self, feats):
        raise NotImplementedError
        
    def get_reference_coords(self, feats, parameters):
        return None

class FlatBottomPotential(Potential):
    def compute_function(self, value, k, lower_bounds, upper_bounds, negation_mask=None, compute_derivative=False):
        if lower_bounds is None:
            lower_bounds = torch.full((value.shape[-1],), float('-inf')).to(value)
        if upper_bounds is None:
            upper_bounds = torch.full((value.shape[-1],), float('inf')).to(value)

        if negation_mask is not None:
            unbounded_below_mask = torch.isneginf(lower_bounds)
            unbounded_above_mask = torch.isposinf(upper_bounds)
            unbounded_mask = unbounded_below_mask + unbounded_above_mask
            assert torch.all(unbounded_mask + negation_mask)
            lower_bounds[~unbounded_above_mask * ~negation_mask] = upper_bounds[~unbounded_above_mask * ~negation_mask]
            upper_bounds[~unbounded_above_mask * ~negation_mask] = float('inf')
            upper_bounds[~unbounded_below_mask * ~negation_mask] = lower_bounds[~unbounded_below_mask * ~negation_mask]
            lower_bounds[~unbounded_below_mask * ~negation_mask] = float('-inf')

        neg_overflow_mask = value < lower_bounds
        pos_overflow_mask = value > upper_bounds

        energy = torch.zeros_like(value)
        energy[neg_overflow_mask] = (k * (lower_bounds - value))[neg_overflow_mask]
        energy[pos_overflow_mask] = (k * (value - upper_bounds))[pos_overflow_mask]
        if not compute_derivative:
            return energy

        dEnergy = torch.zeros_like(value)
        dEnergy[neg_overflow_mask] = -1 * k.expand_as(neg_overflow_mask)[neg_overflow_mask]
        dEnergy[pos_overflow_mask] = 1 * k.expand_as(pos_overflow_mask)[pos_overflow_mask]

        return energy, dEnergy

class ReferencePotential(Potential):
    def compute_variable(self, coords, index, ref_coords, compute_gradient=False):
        ref_mask = torch.zeros(coords.shape[-2], dtype=torch.float32, device=coords.device)
        ref_mask[index[0]] = 1.0
        ref_mask = ref_mask.repeat(coords.shape[0],1)
        aligned_ref_coords = weighted_rigid_align(
            ref_coords.repeat(coords.shape[0],1,1).float(),
            coords.float(),
            ref_mask,
            ref_mask,
        )

        r = coords.index_select(-2, index[0]) -  aligned_ref_coords.index_select(-2, index[0])
        r_norm = torch.linalg.norm(r, dim=-1)
        r_hat = r / r_norm.unsqueeze(-1)
        grad = r_hat.unsqueeze(1)
        coords_prime = coords

        if not compute_gradient:
            return r_norm

        return r_norm, grad

class DistancePotential(Potential):
    def compute_variable(self, coords, index, ref_coords=None, compute_gradient=False):
        r_ij = coords.index_select(-2, index[0]) -  coords.index_select(-2, index[1])
        r_ij_norm = torch.linalg.norm(r_ij, dim=-1)
        r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)

        if not compute_gradient:
            return r_ij_norm

        grad_i = r_hat_ij
        grad_j = -1 * r_hat_ij
        grad = torch.stack((grad_i, grad_j), dim=1)
        return r_ij_norm, grad

class DihedralPotential(Potential):
    def compute_variable(self, coords, index, ref_coords=None, compute_gradient=False):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])
    
        n_ijk = torch.cross(r_ij, r_kj, dim=-1)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)
        
        r_kj_norm = torch.linalg.norm(r_kj, dim=-1)
        n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1)
        n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1)
        
        sign_phi = torch.sign(r_kj.unsqueeze(-2) @ torch.cross(n_ijk, n_jkl, dim=-1).unsqueeze(-1)).squeeze(-1, -2)
        phi = sign_phi * torch.arccos(
            torch.clamp(
                (n_ijk.unsqueeze(-2) @ n_jkl.unsqueeze(-1)).squeeze(-1,-2) /
                (n_ijk_norm * n_jkl_norm),
                -1 + 1e-8,
                1 - 1e-8
            )
        )

        if not compute_gradient:
            return phi

        a = ((r_ij.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm ** 2)).unsqueeze(-1)
        b = ((r_kl.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm ** 2)).unsqueeze(-1)
        
        grad_i = n_ijk * (r_kj_norm / n_ijk_norm ** 2).unsqueeze(-1)
        grad_l = -1 * n_jkl * (r_kj_norm / n_jkl_norm ** 2).unsqueeze(-1)
        grad_j = (a - 1) * grad_i - b * grad_l
        grad_k = (b - 1) * grad_l - a * grad_i
        grad = torch.stack((grad_i, grad_j, grad_k, grad_l), dim=1)        
        return phi, grad

class AbsDihedralPotential(DihedralPotential):
    def compute_variable(self, coords, index, ref_coords=None, compute_gradient=False):
        if not compute_gradient:
            phi = super().compute_variable(coords, index, compute_gradient=compute_gradient)
            phi = torch.abs(phi)
            return phi

        phi, grad = super().compute_variable(coords, index, compute_gradient=compute_gradient)
        grad[(phi < 0)[...,None,:,None].expand_as(grad)] *= -1
        phi = torch.abs(phi)

        return phi, grad 

class PoseBustersPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats['rdkit_bounds_index'][0]
        lower_bounds = feats['rdkit_lower_bounds'][0].clone()
        upper_bounds = feats['rdkit_upper_bounds'][0].clone()
        bond_mask = feats['rdkit_bounds_bond_mask'][0]
        angle_mask = feats['rdkit_bounds_angle_mask'][0]
        
        lower_bounds[bond_mask * ~angle_mask] *= (1.0 - parameters['bond_buffer'])
        upper_bounds[bond_mask * ~angle_mask] *= (1.0 + parameters['bond_buffer'])
        lower_bounds[~bond_mask * angle_mask] *= (1.0 - parameters['angle_buffer'])
        upper_bounds[~bond_mask * angle_mask] *= (1.0 + parameters['angle_buffer'])
        lower_bounds[bond_mask * angle_mask] *= (1.0 - min(parameters['angle_buffer'], parameters['angle_buffer']))
        upper_bounds[bond_mask * angle_mask] *= (1.0 + min(parameters['angle_buffer'], parameters['angle_buffer']))
        lower_bounds[~bond_mask * ~angle_mask] *= (1.0 - parameters['clash_buffer'])
        upper_bounds[~bond_mask * ~angle_mask] = float('inf')

        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None

    def compute_normalization(self, feats, parameters):
        return feats['rdkit_bounds_index'][0].shape[1]

class ConnectionsPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats['connected_atom_index'][0]
        lower_bounds = None
        upper_bounds = torch.full((pair_index.shape[1],), parameters['buffer'], device=pair_index.device)
        k = torch.ones_like(upper_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None

    def compute_normalization(self, feats, parameters):
        return feats['connected_atom_index'][0].shape[1]

class VDWOverlapPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float())
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats['atom_pad_mask'][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = (chain_sizes > 1)[atom_chain_id]

        vdw_radii = torch.zeros(const.num_elements, dtype=torch.float32, device=atom_chain_id.device)
        vdw_radii[1:119] = torch.tensor(const.vdw_radii, dtype=torch.float32, device=atom_chain_id.device)
        atom_vdw_radii = (feats['ref_element'].float() @ vdw_radii.unsqueeze(-1)).squeeze(-1)[0]

        pair_index = torch.triu_indices(atom_chain_id.shape[0], atom_chain_id.shape[0], 1, device=atom_chain_id.device)
        
        pair_pad_mask = atom_pad_mask[pair_index].all(dim=0)       
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]

        num_chains = atom_chain_id.max() + 1
        connected_chain_index = feats['connected_chain_index'][0]
        connected_chain_matrix = torch.eye(num_chains, device=atom_chain_id.device, dtype=torch.bool)
        connected_chain_matrix[connected_chain_index[0], connected_chain_index[1]] = True
        connected_chain_matrix[connected_chain_index[1], connected_chain_index[0]] = True
        connected_chain_mask = connected_chain_matrix[atom_chain_id[pair_index[0]], atom_chain_id[pair_index[1]]]

        pair_index = pair_index[:, pair_pad_mask * pair_ion_mask * ~connected_chain_mask]

        lower_bounds = atom_vdw_radii[pair_index].sum(dim=0) * (1.0 - parameters['buffer'])
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None

    def compute_normalization(self, feats, parameters):
        return feats['atom_pad_mask'][0].sum()


class SymmetricChainCOMPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float())
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats['atom_pad_mask'][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = chain_sizes > 1

        pair_index = feats['symmetric_chain_index'][0]
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]
        pair_index = pair_index[:, pair_ion_mask]
        lower_bounds = torch.full((pair_index.shape[1],), parameters['buffer'], dtype=torch.float32, device=pair_index.device)
        upper_bounds = None
        k = torch.ones_like(lower_bounds)
    
        return pair_index, (k, lower_bounds, upper_bounds), (atom_chain_id, atom_pad_mask), None,  None

    def compute_normalization(self, feats, parameters):
        return torch.unique(feats["asym_id"]).shape[0]

class StereoBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        stereo_bond_index = feats['stereo_bond_index'][0]
        stereo_bond_orientations = feats['stereo_bond_orientations'][0].bool()
        
        lower_bounds = torch.zeros(stereo_bond_orientations.shape, device=stereo_bond_orientations.device)
        upper_bounds = torch.zeros(stereo_bond_orientations.shape, device=stereo_bond_orientations.device)
        lower_bounds[stereo_bond_orientations] = torch.pi - parameters['buffer']
        upper_bounds[stereo_bond_orientations] = float('inf')
        lower_bounds[~stereo_bond_orientations] = float('-inf')
        upper_bounds[~stereo_bond_orientations] = parameters['buffer']
        
        k = torch.ones_like(lower_bounds)

        return stereo_bond_index, (k, lower_bounds, upper_bounds), None, None,  None

    def compute_normalization(self, feats, parameters):
        return feats['stereo_bond_index'][0].shape[1]

class ChiralAtomPotential(FlatBottomPotential, DihedralPotential):
    def compute_args(self, feats, parameters):
        chiral_atom_index = feats['chiral_atom_index'][0]
        chiral_atom_orientations = feats['chiral_atom_orientations'][0].bool()

        lower_bounds = torch.zeros(chiral_atom_orientations.shape, device=chiral_atom_orientations.device)
        upper_bounds = torch.zeros(chiral_atom_orientations.shape, device=chiral_atom_orientations.device)
        lower_bounds[chiral_atom_orientations] = parameters['buffer']
        upper_bounds[chiral_atom_orientations] = float('inf')
        upper_bounds[~chiral_atom_orientations] = -1 * parameters['buffer']
        lower_bounds[~chiral_atom_orientations] = float('-inf')

        k = torch.ones_like(lower_bounds)
        return chiral_atom_index, (k, lower_bounds, upper_bounds), None, None, None

    def compute_normalization(self, feats, parameters):
        return feats['chiral_atom_index'][0].shape[1]

class PlanarBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        double_bond_index = feats['planar_bond_index'][0].T
        double_bond_improper_index = torch.tensor([
            [1,2,3,0],
            [4,5,0,3],
        ], device=double_bond_index.device).T
        improper_index = double_bond_index[:, double_bond_improper_index].swapaxes(0,1).flatten(start_dim=1)
        lower_bounds = None
        upper_bounds = torch.full((improper_index.shape[1],), parameters['buffer'], device=improper_index.device)
        k = torch.ones_like(upper_bounds)

        return improper_index, (k, lower_bounds, upper_bounds), None, None, None

    def compute_normalization(self, feats, parameters):
        return feats['planar_bond_index'][0].shape[1]

class TemplateReferencePotential(FlatBottomPotential, ReferencePotential):
    def compute_args(self, feats, parameters):
        if 'template_mask_cb' not in feats or feats['template_mask_cb'].shape[1] == 0:
            return None, None, None, None, None

        atom_token_id = (
            torch.bmm(feats["atom_to_token"].float(), feats["token_index"].unsqueeze(-1).float())
            .squeeze(-1)
            .long()
        )[0]
        template_idx = parameters['template_idx']
        cb_atom_idx = torch.argmax(feats['token_to_rep_atom'][template_idx], dim=1)
        index = torch.argwhere(feats['template_mask_cb'][0, template_idx]).T
        upper_bounds =  torch.full(index.shape, parameters['buffer'], device=index.device)
        lower_bounds = None
        k = torch.ones_like(upper_bounds)
        return index, (k, lower_bounds, upper_bounds), None, (atom_token_id, cb_atom_idx), None

    def get_reference_coords(self, feats, parameters):
        template_idx = parameters['template_idx']
        ref_coords = feats['template_cb'][0, template_idx].clone()
        return ref_coords

    def compute_normalization(self, feats, parameters):
        template_idx = parameters['template_idx']
        return feats['template_mask_cb'][0, template_idx].sum()

class ContactPotentital(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        index = feats['contact_pair_index'][0]
        union_index = feats['contact_union_index'][0]
        negation_mask = feats['contact_negation_mask'][0]
        lower_bounds = None
        upper_bounds = feats['contact_thresholds'][0].clone()
        k = torch.ones_like(upper_bounds)
        return index, (k, lower_bounds, upper_bounds), None, None, (negation_mask, union_index)

    def compute_normalization(self, feats, parameters):
        return feats['contact_union_index'][0].max() + 1

def get_potentials(boltz2=False):
    potentials = [
        SymmetricChainCOMPotential(
            parameters={
                'guidance_interval': 4,
                'guidance_weight': 0.5,
                'resampling_weight': 1.0,
                'buffer': ExponentialInterpolation(
                    start=1.0,
                    end=5.0,
                    alpha=-2.0
                )
            }
        ),
        VDWOverlapPotential(
            parameters={
                'guidance_interval': 5,
                'guidance_weight': PiecewiseStepFunction(
                    thresholds=[0.4],
                    values=[0.125, 0.0]
                ),
                'resampling_weight': PiecewiseStepFunction(
                    thresholds=[0.6],
                    values=[10.0, 0.0]
                ),
                'buffer': 0.225,
            }
        ),
        ConnectionsPotential(
            parameters={
                'guidance_interval': 1,
                'guidance_weight': 0.15,
                'resampling_weight': 1.0,
                'buffer': 2.0,
            }
        ),
        PoseBustersPotential(
            parameters={
                'guidance_interval': 1,
                'guidance_weight': 0.01,
                'resampling_weight': 1.0,
                'bond_buffer': 0.2,
                'angle_buffer': 0.2,
                'clash_buffer': 0.15
            }
        ),
        ChiralAtomPotential(
            parameters={
                'guidance_interval': 1,
                'guidance_weight': 0.1,
                'resampling_weight': 1.0,
                'buffer': 0.61547
            }
        ),
        StereoBondPotential(
            parameters={
                'guidance_interval': 1,
                'guidance_weight': 0.05,
                'resampling_weight': 1.0,
                'buffer': 0.26180
            }
        ),
        PlanarBondPotential(
            parameters={
                'guidance_interval': 1,
                'guidance_weight': 0.05,
                'resampling_weight': 1.0,
                'buffer': 0.26180
            }
        )
    ]
    if boltz2:
        potentials.extend([
            ContactPotentital(
                parameters={
                    'guidance_interval': 5,
                    'guidance_weight': PiecewiseStepFunction(
                        thresholds=[0.75],
                        values=[0.0, 0.25]
                    ),
                    'resampling_weight': 1.0,
                    'union_lambda': ExponentialInterpolation(
                        start=10.0,
                        end=0.0,
                        alpha=-2.0
                    )
                }
            ),
            TemplateReferencePotential(
                parameters={
                    'guidance_interval': 1,
                    'guidance_weight': 0.05,
                    'resampling_weight': 1.0,
                    'buffer': 0.5,
                    'template_idx': 0
                }
            )
        ])
    return potentials