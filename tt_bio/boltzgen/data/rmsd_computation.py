from torch import Tensor
from typing import Dict
import torch

from tt_bio.data.mol import minimum_lddt_symmetry_coords
from tt_bio.boltzgen.model.geometry import weighted_minimum_rmsd


def get_true_coordinates(
    batch: Dict[str, Tensor],
    out: Dict[str, Tensor],
    diffusion_samples: int,
    symmetry_correction: bool,
    expand_to_diffusion_samples: bool = True,
    protein_lig_rmsd=False,
):
    if protein_lig_rmsd:
        assert not symmetry_correction, "Not implemented yet"

    if symmetry_correction:
        msg = "expand_to_diffusion_samples must be true for symmetry correction."
        assert expand_to_diffusion_samples, msg

    return_dict = {}

    if (
        symmetry_correction
    ):
        K = batch["coords"].shape[1]
        assert K == 1, (
            f"Symmetry correction is not supported for num_ensembles_val={K}."
        )

        assert batch["coords"].shape[0] == 1, (
            f"Validation is not supported for batch sizes={batch['coords'].shape[0]}"
        )

        true_coords = []
        true_coords_resolved_mask = []
        for idx in range(batch["token_index"].shape[0]):
            for rep in range(diffusion_samples):
                i = idx * diffusion_samples + rep

                best_true_coords, best_true_coords_resolved_mask = (
                    minimum_lddt_symmetry_coords(
                        coords=out["sample_atom_coords"][i : i + 1],
                        feats=batch,
                        index_batch=idx,
                    )
                )

                true_coords.append(best_true_coords)
                true_coords_resolved_mask.append(best_true_coords_resolved_mask)

        true_coords = torch.cat(true_coords, dim=0)
        true_coords_resolved_mask = torch.cat(true_coords_resolved_mask, dim=0)
        true_coords = true_coords.unsqueeze(1)


        return_dict["true_coords"] = true_coords
        return_dict["true_coords_resolved_mask"] = true_coords_resolved_mask
        return_dict["rmsds"] = 0
        return_dict["best_rmsd_recall"] = 0

    else:
        assert batch["coords"].shape[0] == 1, (
            f"Validation is not supported for batch sizes={batch['coords'].shape[0]}"
        )
        K, L = batch["coords"].shape[1:3]

        true_coords_resolved_mask = batch["atom_resolved_mask"] 
        true_coords = batch["coords"].squeeze(0)
        if expand_to_diffusion_samples:
            true_coords = true_coords.repeat((diffusion_samples, 1, 1)).reshape(
                diffusion_samples, K, L, 3
            )

            true_coords_resolved_mask = true_coords_resolved_mask.repeat_interleave(
                diffusion_samples, dim=0
            )  # since all masks are the same across conformers and diffusion samples, can just repeat S times
        else:
            true_coords_resolved_mask = true_coords_resolved_mask.squeeze(0)

        return_dict["true_coords"] = true_coords
        return_dict["true_coords_resolved_mask"] = true_coords_resolved_mask
        return_dict["rmsds"] = 0
        return_dict["best_rmsd_recall"] = 0
        return_dict["best_rmsd_precision"] = 0

        if protein_lig_rmsd:
            (
                rmsd,
                best_rmsd,
                rmsd_design,
                best_rmsd_design,
                rmsd_target,
                best_rmsd_target,
                rmsd_design_target,
                best_rmsd_design_target,
                target_aligned_rmsd_design,
                best_target_aligned_rmsd_design,
            ) = weighted_minimum_rmsd(
                out["sample_atom_coords"],
                batch,
                multiplicity=diffusion_samples,
                protein_lig_rmsd=protein_lig_rmsd,
            )
            return_dict["rmsd"] = rmsd
            return_dict["best_rmsd"] = best_rmsd
            return_dict["rmsd_design"] = rmsd_design
            return_dict["best_rmsd_design"] = best_rmsd_design
            return_dict["rmsd_target"] = rmsd_target
            return_dict["best_rmsd_target"] = best_rmsd_target
            return_dict["rmsd_design_target"] = rmsd_design_target
            return_dict["best_rmsd_design_target"] = best_rmsd_design_target
            return_dict["target_aligned_rmsd_design"] = target_aligned_rmsd_design
            return_dict["best_target_aligned_rmsd_design"] = best_target_aligned_rmsd_design
    return return_dict
