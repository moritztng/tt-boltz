"""Inference-time geometry helpers.

Survivors from the upstream ``model/loss/`` directory. The original was
~2.8k LOC of training losses (confidence/distogram/bfactor/res_type/validation
+ smooth-lddt + bond-loss); inference only needs five rigid-alignment / lddt
helpers used by the diffusion sampler, the writer (for symmetry/RMSD), and
the analyze stage.
"""
from typing import Dict

import torch
from einops import einsum

from tt_boltz.data import const


def weighted_rigid_align(
    true_coords,
    pred_coords,
    weights,
    mask,
):
    """Kabsch-style weighted alignment (AF3 Algorithm 28).

    Aligns ``true_coords`` to ``pred_coords`` under per-atom ``weights`` and
    ``mask``. Returns the aligned true coords with ``requires_grad=False``.
    """
    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights).unsqueeze(-1)

    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if torch.any(mask.sum(dim=-1) < (dim + 1)):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            "`WeightedRigidAlign` cannot return a unique rotation."
        )

    cov_matrix = einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)

    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    )
    V = V.mH

    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print(
            "Warning: Excessively low rank of cross-correlation between aligned "
            "point clouds. `WeightedRigidAlign` cannot return a unique rotation."
        )

    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)

    F = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device)[
        None
    ].repeat(batch_size, 1, 1)
    F[:, -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    aligned_coords = (
        einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )
    aligned_coords.detach_()

    return aligned_coords


def lddt_dist(dmat_predicted, dmat_true, mask, cutoff=15.0, per_atom=False):
    """Distance-difference LDDT against a pairwise mask."""
    dists_to_score = (dmat_true < cutoff).float() * mask
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    score = 0.25 * (
        (dist_l1 < 0.5).float()
        + (dist_l1 < 1.0).float()
        + (dist_l1 < 2.0).float()
        + (dist_l1 < 4.0).float()
    )

    if per_atom:
        mask_no_match = torch.sum(dists_to_score, dim=-1) != 0
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=-1))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=-1))
        return score, mask_no_match.float()
    norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=(-2, -1)))
    score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=(-2, -1)))
    total = torch.sum(dists_to_score, dim=(-1, -2))
    return score, total


def factored_lddt_loss(
    true_atom_coords,
    pred_atom_coords,
    feats,
    atom_mask,
    multiplicity=1,
    cardinality_weighted=False,
    exclude_ions=False,
):
    """Per-interaction-type lddt; returns dicts indexed by interaction kind."""
    with torch.autocast("cuda", enabled=False):
        atom_type = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )
        atom_type = atom_type.repeat_interleave(multiplicity, 0)

        chain_id = feats["asym_id"]
        atom_chain_id = (
            torch.bmm(feats["atom_to_token"].float(), chain_id.unsqueeze(-1).float())
            .squeeze(-1)
            .long()
        )
        atom_chain_id = atom_chain_id.repeat_interleave(multiplicity, 0)
        same_chain_mask = (
            atom_chain_id[:, :, None] == atom_chain_id[:, None, :]
        ).float()

        pair_mask = atom_mask[:, :, None] * atom_mask[:, None, :]
        pair_mask = (
            pair_mask
            * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
        )

        ligand_mask = (atom_type == const.chain_type_ids["NONPOLYMER"]).float()
        if exclude_ions:
            ions = (torch.sum(same_chain_mask * pair_mask, dim=-1) == 1).float()
            ligand_mask = ligand_mask * (1 - ions)

    dna_mask = (atom_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (atom_type == const.chain_type_ids["RNA"]).float()
    design_mask = torch.bmm(
        feats["atom_to_token"].float(), feats["design_mask"].float().unsqueeze(-1)
    ).squeeze(-1)
    protein_mask = (atom_type == const.chain_type_ids["PROTEIN"]).float()
    protein_mask = protein_mask * (1 - design_mask)

    nucleotide_mask = dna_mask + rna_mask

    true_d = torch.cdist(true_atom_coords, true_atom_coords)
    pred_d = torch.cdist(pred_atom_coords, pred_atom_coords)

    cutoff = 15 + 15 * (
        1 - (1 - nucleotide_mask[:, :, None]) * (1 - nucleotide_mask[:, None, :])
    )
    del nucleotide_mask

    def _pair(a, b):
        return pair_mask * (a[:, :, None] * b[:, None, :] + b[:, :, None] * a[:, None, :])

    def _intra_diff_chain(a, b):
        return pair_mask * (1 - same_chain_mask) * (a[:, :, None] * b[:, None, :])

    def _intra_same_chain(a, b):
        return pair_mask * same_chain_mask * (a[:, :, None] * b[:, None, :])

    pairs = {
        "design_protein": _pair(design_mask, protein_mask),
        "design_ligand": _pair(design_mask, ligand_mask),
        "design_dna": _pair(design_mask, dna_mask),
        "design_rna": _pair(design_mask, rna_mask),
        "dna_protein": _pair(dna_mask, protein_mask),
        "rna_protein": _pair(rna_mask, protein_mask),
        "ligand_protein": _pair(ligand_mask, protein_mask),
        "dna_ligand": _pair(dna_mask, ligand_mask),
        "rna_ligand": _pair(rna_mask, ligand_mask),
        "intra_dna": pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :]),
        "intra_rna": pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :]),
        "intra_ligand": _intra_same_chain(ligand_mask, ligand_mask),
        "intra_protein": _intra_same_chain(protein_mask, protein_mask),
        "intra_design": _intra_same_chain(design_mask, design_mask),
        "design_design": _intra_diff_chain(design_mask, design_mask),
        "protein_protein": _intra_diff_chain(protein_mask, protein_mask),
    }

    lddt_dict: Dict[str, torch.Tensor] = {}
    total_dict: Dict[str, torch.Tensor] = {}
    for key, m in pairs.items():
        lddt_dict[key], total_dict[key] = lddt_dist(pred_d, true_d, m, cutoff)

    modified_residue = (
        feats["mol_type"] != const.chain_type_ids["NONPOLYMER"]
    ).float() * (
        feats["res_type"][:, :, const.token_ids["UNK"]]
        + feats["res_type"][:, :, const.token_ids["DN"]]
        + feats["res_type"][:, :, const.token_ids["N"]]
    )
    modified_atom_mask = (
        torch.bmm(
            feats["atom_to_token"].float(), modified_residue.unsqueeze(-1).float()
        )
        .squeeze(-1)
        .long()
    )
    modified_atom_mask = modified_atom_mask.repeat_interleave(multiplicity, 0)
    modified_vs_all_mask = pair_mask * modified_atom_mask[:, :, None]
    lddt_dict["modified"], total_dict["modified"] = lddt_dist(
        pred_d, true_d, modified_vs_all_mask, cutoff
    )

    if not cardinality_weighted:
        for key in total_dict:
            total_dict[key] = (total_dict[key] > 0.0).float()

    return lddt_dict, total_dict


def compute_subset_rmsd(
    atom_coords,
    pred_atom_coords,
    atom_mask,
    align_weights,
    subset_mask,
    multiplicity,
    rmsd_mask=None,
):
    """Align on ``subset_mask`` and report rmsd over ``rmsd_mask`` (default = subset)."""
    used_mask = atom_mask * subset_mask
    used_weights = align_weights * subset_mask

    if used_mask.sum() == 0:
        return torch.tensor(torch.nan), torch.tensor(torch.nan)

    with torch.no_grad():
        aligned_coords = weighted_rigid_align(
            atom_coords, pred_atom_coords, used_weights, mask=used_mask
        )

    mse = ((pred_atom_coords - aligned_coords) ** 2).sum(dim=-1)
    rmsd_mask = used_weights * used_mask if rmsd_mask is None else rmsd_mask * atom_mask
    rmsd = torch.sqrt(
        torch.sum(mse * rmsd_mask, dim=-1)
        / torch.sum(rmsd_mask, dim=-1).clamp_min(1e-7)
    )
    best_rmsd = torch.min(rmsd.reshape(-1, multiplicity), dim=1).values
    return rmsd, best_rmsd


def weighted_minimum_rmsd(
    pred_atom_coords,
    feats,
    multiplicity=1,
    nucleotide_weight=5.0,
    ligand_weight=10.0,
    representative_atoms=False,
    protein_lig_rmsd=False,
):
    """Best rmsd across diffusion samples + reference-ensemble conformers.

    With ``protein_lig_rmsd=True`` also returns design-only / target-only /
    design-aligned / target-aligned breakdowns.
    """
    with torch.autocast("cuda", enabled=False):
        atom_coords = feats["coords"]
        B, K = atom_coords.shape[0:2]
        assert B == 1, "Validation is not supported for batch size > 1"
        atom_coords = atom_coords.squeeze(0).repeat((multiplicity, 1, 1))
        pred_atom_coords = pred_atom_coords.repeat_interleave(K, 0)

        if representative_atoms:
            atom_mask = feats["token_to_rep_atom"].sum(dim=1)
        else:
            atom_mask = feats["atom_resolved_mask"]
        atom_mask = atom_mask.repeat_interleave(K * multiplicity, 0)

        align_weights = atom_coords.new_ones(atom_coords.shape[:2])
        atom_type = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )
        atom_type = atom_type.repeat_interleave(K * multiplicity, 0)

        align_weights = align_weights * (
            1
            + nucleotide_weight
            * (
                torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
            )
            + ligand_weight
            * torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()
        )

        with torch.no_grad():
            atom_coords_aligned_ground_truth = weighted_rigid_align(
                atom_coords, pred_atom_coords, align_weights, mask=atom_mask
            )

    mse_loss = ((pred_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(dim=-1)
    rmsd = torch.sqrt(
        torch.sum(mse_loss * align_weights * atom_mask, dim=-1)
        / torch.sum(align_weights * atom_mask, dim=-1)
    )
    best_rmsd = torch.min(rmsd.reshape(-1, multiplicity), dim=1).values

    if not protein_lig_rmsd:
        return rmsd, best_rmsd

    design_mask = torch.bmm(
        feats["atom_to_token"].float(), feats["design_mask"].float().unsqueeze(-1)
    ).squeeze(-1)
    design_mask = design_mask.repeat_interleave(multiplicity, 0)
    design_chain_mask = torch.bmm(
        feats["atom_to_token"].float(), feats["chain_design_mask"].float().unsqueeze(-1)
    ).squeeze(-1)
    design_chain_mask = design_chain_mask.repeat_interleave(multiplicity, 0)
    target_mask = 1.0 - design_chain_mask

    def _safe(name, *args, **kwargs):
        try:
            return compute_subset_rmsd(*args, **kwargs)
        except Exception as e:
            print(f"Warning: {name} failed with error: {e}")
            return torch.tensor(torch.nan), torch.tensor(torch.nan)

    rmsd_design, best_rmsd_design = _safe(
        "rmsd_design", atom_coords, pred_atom_coords, atom_mask, align_weights,
        design_mask, multiplicity,
    )
    rmsd_target, best_rmsd_target = _safe(
        "rmsd_target", atom_coords, pred_atom_coords, atom_mask, align_weights,
        target_mask, multiplicity,
    )

    rmsd_design_target = torch.tensor(torch.nan)
    best_rmsd_design_target = torch.tensor(torch.nan)
    target_aligned_rmsd_design = torch.tensor(torch.nan)
    best_target_aligned_rmsd_design = torch.tensor(torch.nan)

    if (
        torch.any(
            feats["mol_type"][~feats["chain_design_mask"]]
            != const.chain_type_ids["NONPOLYMER"]
        )
        or (~feats["chain_design_mask"]).float().sum() > 3
    ):
        rmsd_design_target, best_rmsd_design_target = _safe(
            "rmsd_design_target",
            atom_coords, pred_atom_coords, atom_mask, align_weights,
            atom_mask, multiplicity, rmsd_mask=design_mask,
        )
        target_aligned_rmsd_design, best_target_aligned_rmsd_design = _safe(
            "target_aligned_rmsd_design",
            atom_coords, pred_atom_coords, atom_mask, target_mask,
            atom_mask, multiplicity, rmsd_mask=design_mask,
        )

    return (
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
    )
