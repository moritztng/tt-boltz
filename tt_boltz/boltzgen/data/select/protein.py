from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist

from tt_boltz.data import const
from tt_boltz.boltzgen.data.select.selector import Selector
from tt_boltz.boltzgen.data.crop.multimer import MultimerCropper
from tt_boltz.boltzgen.data.data import (
    Atom,
    Tokenized,
    TokenBond,
)


def min_token_distances(
    tokens1: Tokenized,
    tokens2: Tokenized,
    random: np.random.Generator,
    noise_std: float = 1,
    axis: int = 1,
):
    tokens2_centers = tokens2["center_coords"].copy()
    tokens2_centers[~tokens2["resolved_mask"]] = np.nan
    tokens1_centers = tokens1["center_coords"].copy()
    tokens1_centers[~tokens1["resolved_mask"]] = np.nan
    return min_distances(tokens1_centers, tokens2_centers, random, noise_std, axis)


def min_distances(
    coords1: np.ndarray,
    coords2: np.ndarray,
    random: np.random.Generator,
    noise_std: float = 1,
    axis: int = 1,
):
    distances = cdist(coords1, coords2)
    distances[np.isnan(distances)] = np.inf
    min_distances = np.min(distances, axis=axis)
    noisy_distances = (
        min_distances + random.normal(size=min_distances.shape) * noise_std
    )
    return noisy_distances


class ProteinSelector(Selector):
    """Select design tokens from protein chains."""

    def __init__(
        self,
        design_neighborhood_sizes: List[int] = [10],
        substructure_neighborhood_sizes: List[int] = [10],
        distance_noise_std: float = 1,
        run_selection: bool = True,
        specify_binding_sites: bool = False,
        select_all: bool = False,
        complete_structure_mask: bool = False,
        binding_token_cutoff: float = 15,
        binding_atom_cutoff: float = 5,
        max_msa_prob: float = 0.6,
        min_msa_prob: float = 0.1,
        target_msa_sampling_length_cutoff: int = 50,
        structure_condition_prob: float = 0,
        ss_condition_prob: float = 0,
        binding_site_probs: Optional[Dict[str, float]] = None,
        structure_probs: Optional[Dict[str, float]] = None,
        chain_reindexing=False,
        moldir: str = None,
        simple_selection=False,
    ) -> None:
        """Initialize the selector.

        Parameters
        ----------
        neighborhood_sizes : List[int]
            Modulates the type of selection to be performed.
            TODO: write doc

        """
        self.design_neighborhood_sizes = design_neighborhood_sizes
        self.substructure_neighborhood_sizes = substructure_neighborhood_sizes
        self.cropper = MultimerCropper(design_neighborhood_sizes)
        self.distance_noise_std = distance_noise_std
        self.run_selection = run_selection
        self.select_all = select_all
        self.specify_binding_sites = specify_binding_sites
        self.binding_token_cutoff = binding_token_cutoff
        self.binding_atom_cutoff = binding_atom_cutoff
        self.max_msa_prob = max_msa_prob
        self.min_msa_prob = min_msa_prob
        self.ss_condition_prob = ss_condition_prob
        self.target_msa_sampling_length_cutoff = target_msa_sampling_length_cutoff

        self.selection_functions = {
            "select_none": self.select_none,
            "select_scaffold": self.select_scaffold,
            "select_motif": self.select_motif,
            "select_scaffold_binder": self.select_scaffold_binder,
            "select_motif_binder": self.select_motif_binder,
            "select_nonprot_interface": self.select_nonprot_interface,
            "select_standard_prot": self.select_standard_prot,
            "select_protein_intefaces": self.select_protein_intefaces,
            "select_protein_chains": self.select_protein_chains,
        }
        self.probabilities = (
            const.training_task_probabilities_with_reindexing
            if chain_reindexing
            else const.training_task_probabilities_simple
            if simple_selection
            else const.training_task_probabilities
        )
        self.moldir = moldir

        if binding_site_probs is None:
            binding_site_probs = {
                "specify_binding": 0.15,
                "specify_not_binding": 0.075,
                "specify_binding_not_binding": 0.075,
                "specify_none": 0.7,
            }
        self.binding_type_tasks = [
            (prob, getattr(self, name)) for name, prob in binding_site_probs.items()
        ]

        if structure_probs is None:
            structure_probs = {
                "structure_all": 0.4,
                "structure_uniform": 0.3,
                "structure_crops": 0.3,
            }
        self.structure_tasks = [
            (prob, getattr(self, name)) for name, prob in structure_probs.items()
        ]

        self.ss_condition_tasks = [
            (0.5, self.ss_all),
            (0.5, self.ss_uniform),
        ]

        self.structure_condition_prob = structure_condition_prob
        self.complete_structure_mask = complete_structure_mask

    def select(  # noqa: PLR0915
        self,
        data: Tokenized,
        random: np.random.Generator,
    ) -> Tokenized:
        """Select protein residues to be designed.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        random : np.random.Generator
            The random state for reproducibility.

        Returns
        -------
        Tokenized
            The cropped data.

        """
        if not self.run_selection:
            return data, "predict"

        # Get token data
        tokens = data.tokens.copy()
        prot_mask = tokens["mol_type"] == const.chain_type_ids["PROTEIN"]
        standard_mask = tokens["is_standard"].astype(bool)

        # Atomized protein tokens are always predicted and never designed
        # However, we never use them as design targets
        atomized_prot_tokens = tokens[prot_mask & ~standard_mask]
        prot_tokens = tokens[prot_mask & standard_mask]
        nonprot_tokens = tokens[~prot_mask]

        # Get chains
        prot_chain_ids = np.unique(prot_tokens["asym_id"])
        num_prot_chains = len(prot_chain_ids)
        nonprot_chain_ids = np.unique(nonprot_tokens["asym_id"])
        num_nonprot_chains = len(nonprot_chain_ids)

        # Get selection distribution
        if self.select_all:
            task = "select_all"
        elif num_prot_chains == 0:
            task = "0prot_>=0nonprot"
        elif num_prot_chains == 1 and num_nonprot_chains == 0:
            task = "1prot_0nonprot"
        elif num_prot_chains == 1 and num_nonprot_chains > 0:
            task = "1prot_>0nonprot"
        elif num_prot_chains > 1 and num_nonprot_chains == 0:
            task = ">1prot_0nonprot"
        elif num_prot_chains > 1 and num_nonprot_chains > 0:
            task = ">1prot_>0nonprot"
        else:
            raise NotImplementedError

        # Select "design_mask" feature in the tokens
        task_distribution = self.probabilities[task]
        weights, functions = zip(*task_distribution)
        selection_fn = self.selection_functions[random.choice(functions, p=weights)]
        tokens = selection_fn(tokens, random)

        # Reset token_idx and token bonds
        old_indices = tokens["token_idx"]
        old_to_new = {old: new for new, old in enumerate(old_indices)}
        token_bonds = data.bonds
        token_bonds = np.array(
            [
                (old_to_new[bond["token_1"]], old_to_new[bond["token_2"]], bond["type"])
                for bond in token_bonds
                if bond["token_1"] in old_indices and bond["token_2"] in old_indices
            ],
            dtype=TokenBond,
        )
        tokens["token_idx"] = np.arange(len(tokens))

        # Select "binding_type" conditioning feature
        self.run_specification(tokens, random, data.structure.atoms)

        # Construct the token_distance_mask for conditioning on distances
        self.run_distance_sampling(tokens, random)

        # Sample whether to keep MSAs
        self.run_target_msa_sampling(tokens, random)

        # Sample whether to give the secondary structure as input
        self.run_ss_mask_specification(tokens, random)

        tokenized_selected = replace(data, tokens=tokens, bonds=token_bonds)
        return tokenized_selected, task + str(selection_fn)

    def run_ss_mask_specification(
        self, tokens: np.ndarray, random: np.random.Generator
    ):
        design_mask = tokens["design_mask"].astype(bool)
        if design_mask.sum() > 1 and random.random() < self.ss_condition_prob:
            weights, functions = zip(*self.ss_condition_tasks)
            ss_fn = random.choice(functions, p=weights)
            ss_fn(tokens, random)

    def ss_all(self, tokens, random):
        design_mask = tokens["design_mask"].astype(bool)
        tokens["design_ss_mask"][design_mask] = 1

    def ss_uniform(self, tokens, random):
        design_mask = tokens["design_mask"].astype(bool)
        num_sets = random.integers(1, len(tokens))
        split_points = sorted(
            random.choice(range(1, len(tokens)), num_sets, replace=False)
        )

        start = 0
        for end in split_points:
            if random.random() > 0.5:
                interval_mask = np.zeros_like(design_mask).astype(bool)
                interval_mask[start:end] = True
                tokens["design_ss_mask"][design_mask & interval_mask] = 1
            start = end
        if random.random() > 0.5:
            interval_mask = np.zeros_like(design_mask).astype(bool)
            interval_mask[start:end] = True
            tokens["design_ss_mask"][design_mask & interval_mask] = 1

    def run_target_msa_sampling(self, tokens: np.ndarray, random: np.random.Generator):
        design_mask = tokens["design_mask"]
        # compute length of targets
        diff = np.diff(design_mask, prepend=design_mask[0], append=design_mask[-1])
        target_msa_mask = np.zeros_like(design_mask)
        # find start of targetsd
        starts = np.concatenate(([0], np.where(diff != 0)[0]))
        lengths = np.diff(starts, append=len(design_mask))
        for i in range(len(starts)):
            seq_idx, l = starts[i], lengths[i]
            if design_mask[seq_idx] == 1:
                target_msa_mask[seq_idx : seq_idx + l] = (
                    1  # designed parts never have MSAs
                )
                continue
            if l > self.target_msa_sampling_length_cutoff:
                keep_msa_prob = self.max_msa_prob
            else:
                keep_msa_prob = (
                    (self.max_msa_prob - self.min_msa_prob)
                    * (l - 1)
                    / self.target_msa_sampling_length_cutoff
                )
            keep_msa = 1 - int(random.random() <= keep_msa_prob)
            target_msa_mask[seq_idx : seq_idx + l] = keep_msa
        tokens["target_msa_mask"] = target_msa_mask

    def run_distance_sampling(self, tokens: np.ndarray, random: np.random.Generator):
        if self.complete_structure_mask:
            tokens["structure_group"] = 1
            return

        design_mask = tokens["design_mask"].astype(bool)
        resolved_mask = tokens["resolved_mask"].astype(bool)
        target_tokens = tokens[~design_mask & resolved_mask]

        # Sample for which chains to specify the structure
        target_chain_ids = np.unique(target_tokens["asym_id"])
        if (
            len(target_chain_ids) == 0
            or random.random() > self.structure_condition_prob
        ):
            return
        num_specified = random.integers(1, len(target_chain_ids) + 1)
        specified_chains = random.choice(target_chain_ids, num_specified, replace=False)

        # For each chain for which we chose to specify the structure, select all tokens or sub-regions for which to specify the structure.
        substructure_sets = []
        for spec_chain in specified_chains:
            spec_tokens = tokens[(tokens["asym_id"] == spec_chain) & ~design_mask]
            weights, functions = zip(*self.structure_tasks)
            structure_fn = random.choice(functions, p=weights)

            substructure_sets.extend(structure_fn(spec_tokens, random))

        # Sample the number of coodinate systems (we call them groups) into which we will put the motifs. Shift by one because of 1 indexing in the groups (0 corresponds to no structure specification).
        num_groups = random.integers(1, len(substructure_sets) + 1)

        # Get group/frame assigments. Group 0 corresponds to no distance assignment. The other groups are indexed from 1. So there are always at least 2 groups here (0 and 1). If there is no structure conditioning, then there is only one group.
        for substructure_set in substructure_sets:
            group_assignment = random.choice(np.arange(num_groups)) + 1
            tokens["structure_group"][substructure_set["token_idx"]] = group_assignment

        assert tokens["structure_group"][tokens["design_mask"].astype(bool)].sum() == 0

    def structure_all(self, tokens: np.ndarray, random: np.random.Generator):
        return [tokens]

    def structure_uniform(self, tokens: np.ndarray, random: np.random.Generator):
        if len(tokens) == 1:
            return [tokens]
        num_sets = random.integers(1, min(len(tokens), 6))
        split_points = sorted(
            random.choice(range(1, len(tokens)), num_sets, replace=False)
        )

        structure_sets = []
        start = 0
        for end in split_points:
            structure_sets.append(tokens[start:end])
            start = end
        structure_sets.append(tokens[start:])
        return structure_sets

    def structure_crops(self, tokens: np.ndarray, random: np.random.Generator):
        if len(tokens) < 5:
            return [tokens]
        num_substructures = random.integers(2, 5)

        # Create the substructures by keeping track of remaining indices that are not yet included in a subsstructure. The maximum size of the next crop is always sampled between 1 and the number of remaining indices
        substructures = []
        remaining = tokens["token_idx"].copy()
        for _ in range(num_substructures):
            neighborhood_size = random.choice(self.substructure_neighborhood_sizes)

            # Make sure that there are enough indices to select
            if len(remaining) < neighborhood_size + 1:
                break
            num_crop = max(random.integers(len(remaining)), neighborhood_size + 1)

            remaining_tokens = tokens[np.isin(tokens["token_idx"], remaining)]
            query = remaining_tokens[random.integers(len(remaining_tokens))]
            crop_indices = self.cropper.select_cropped_indices(
                tokens=remaining_tokens,
                valid_tokens=remaining_tokens[remaining_tokens["resolved_mask"]],
                query=query,
                neighborhood_size=neighborhood_size,
                max_atoms=num_crop * 10,
                max_tokens=num_crop,
            )

            if len(crop_indices) == 0:
                # This can happen if there are multiple tokens with the same residue index because the max_tokens does not actually correspond to max_tokens but to maximum number of residues
                break

            substructure_tokens = remaining_tokens[crop_indices]
            substructures.append(substructure_tokens)
            remaining = remaining[~np.isin(remaining, substructure_tokens["token_idx"])]

        # Handle edge case that we broke the loop because there werer not enough indices to select from.
        if len(substructures) == 0:
            return [tokens]

        # Assert to check that there are no ovelaps in the substructures
        all_substructured = np.concatenate(substructures)
        assert len(all_substructured) == len(
            np.unique(all_substructured["token_idx"])
        ), (
            "There are overlaps in the substructures during structure conditioning selection."
        )

        return substructures

    def run_specification(
        self, tokens: np.ndarray, random: np.random.Generator, all_atoms: Atom
    ):
        """In place operation to specify the binding_type feature."""

        design_mask = tokens["design_mask"].astype(bool)
        resolved_mask = tokens["resolved_mask"].astype(bool)
        design_tokens = tokens[design_mask & resolved_mask]
        target_tokens = tokens[~design_mask & resolved_mask]

        if (
            not self.specify_binding_sites
            or len(target_tokens) == 0
            or len(design_tokens) == 0
        ):
            return

        # Find binder and target tokens within self.binding_token_cutoff of each other
        target_min_distances = min_token_distances(
            target_tokens, design_tokens, random, self.distance_noise_std
        )
        target_subset = target_tokens[target_min_distances < self.binding_token_cutoff]

        design_min_distances = min_token_distances(
            design_tokens, target_tokens, random, self.distance_noise_std
        )
        design_subset = design_tokens[design_min_distances < self.binding_token_cutoff]

        if len(target_subset) == 0 or len(design_subset) == 0:
            return

        # Get atoms of the tokens that are close to each other
        target_atoms = []
        target_atom_to_token = []
        for idx, t in enumerate(target_subset):
            atoms = all_atoms[t["atom_idx"] : t["atom_idx"] + t["atom_num"]]
            atoms = atoms[atoms["is_present"].astype(bool)]
            target_atoms.append(atoms)
            target_atom_to_token.append([idx] * len(atoms))
        target_atoms = np.concatenate(target_atoms)
        target_atom_to_token = np.concatenate(target_atom_to_token)

        design_atoms = []
        design_atom_to_token = []
        for idx, t in enumerate(design_subset):
            atoms = all_atoms[t["atom_idx"] : t["atom_idx"] + t["atom_num"]]
            atoms = atoms[atoms["is_present"].astype(bool)]
            design_atoms.append(atoms)
            design_atom_to_token.append([idx] * len(atoms))
        design_atoms = np.concatenate(design_atoms)
        design_atom_to_token = np.concatenate(design_atom_to_token)

        # Compute contacts based on atom level distances
        distances = min_distances(
            target_atoms["coords"],
            design_atoms["coords"],
            random,
            self.distance_noise_std,
        )
        target_contacts = target_subset[
            target_atom_to_token[distances < self.binding_atom_cutoff]
        ]
        contact_mask = np.isin(tokens["token_idx"], target_contacts["token_idx"])

        weights, functions = zip(*self.binding_type_tasks)
        binding_fn = random.choice(functions, p=weights)
        binding_fn(tokens, contact_mask, design_mask, random)
        assert tokens["binding_type"][tokens["design_mask"].astype(bool)].sum() == 0

    def specify_none(
        self,
        tokens: np.ndarray,
        contact_mask: np.ndarray,
        design_mask: np.ndarray,
        random: np.random.Generator,
    ):
        pass

    def specify_binding(
        self,
        tokens: np.ndarray,
        contact_mask: np.ndarray,
        design_mask: np.ndarray,
        random: np.random.Generator,
    ):
        assert (contact_mask & design_mask).sum() == 0
        if (contact_mask).sum() == 0:
            return
        elif (contact_mask).sum() == 1:
            num_specified = 1
        else:
            num_specified = random.integers(1, contact_mask.sum())

        specified_idx = random.choice(
            np.arange(len(contact_mask))[contact_mask], num_specified, replace=False
        )
        tokens["binding_type"][specified_idx] = const.binding_type_ids["BINDING"]
        assert tokens["binding_type"][tokens["design_mask"].astype(bool)].sum() == 0

    def specify_not_binding(
        self,
        tokens: np.ndarray,
        contact_mask: np.ndarray,
        design_mask: np.ndarray,
        random: np.random.Generator,
    ):
        not_binding_mask = ~contact_mask & ~design_mask
        if (not_binding_mask).sum() == 0:
            return
        elif (not_binding_mask).sum() == 1:
            num_specified = 1
        else:
            num_specified = random.integers(1, (not_binding_mask).sum())

        specified_idx = random.choice(
            np.arange(len(not_binding_mask))[not_binding_mask],
            num_specified,
            replace=False,
        )
        tokens["binding_type"][specified_idx] = const.binding_type_ids["NOT_BINDING"]
        assert tokens["binding_type"][tokens["design_mask"].astype(bool)].sum() == 0

    def specify_binding_not_binding(
        self,
        tokens: np.ndarray,
        contact_mask: np.ndarray,
        design_mask: np.ndarray,
        random: np.random.Generator,
    ):
        self.specify_binding(tokens, contact_mask, design_mask, random)
        self.specify_not_binding(tokens, contact_mask, design_mask, random)

    def select_none(
        self,
        tokens: np.ndarray,
        random: np.random.Generator,
    ):
        return tokens

    def resect_and_reindex(
        self,
        tokens: np.ndarray,
        random: np.random.Generator,
    ):
        deltas = np.concatenate(
            (tokens["asym_id"].astype(int), np.array([-1]))
        ) - np.concatenate((np.array([-1]), tokens["asym_id"].astype(int)))
        chain_breaks = np.where(deltas != 0)[0]
        deltas = np.concatenate(
            (tokens["design_mask"].astype(int), np.array([-1]))
        ) - np.concatenate((np.array([-1]), tokens["design_mask"].astype(int)))
        design_breaks = np.where(deltas != 0)[0]
        breaks = [
            int(i) for i in sorted(list(set(chain_breaks).union(set(design_breaks))))
        ]
        resection_mask = np.ones_like(tokens["design_mask"])
        for i in range(len(breaks) - 1):
            s, e = breaks[i], breaks[i + 1]
            tokens["feature_asym_id"][s:e] = i
            tokens["feature_res_idx"][s:e] = np.arange(e - s)

            # resect a random number of residues around the boundaries
            if i > 0:
                _s = max(0, s - random.integers(1, 5))
                _e = min(len(resection_mask), s + random.integers(1, 5))
                resection_mask[_s:_e] = 0
        resection_mask = np.clip(
            resection_mask + tokens["design_mask"], a_min=0, a_max=1
        )  # keep all design tokens
        cropped = np.where(resection_mask)[0]
        tokens = tokens[cropped]
        return tokens

    def select_motif(
        self,
        tokens: np.ndarray,
        random: np.random.Generator,
        fixed_crop: bool = False,
    ):
        prot_mask = tokens["mol_type"] == const.chain_type_ids["PROTEIN"]
        standard_mask = tokens["is_standard"].astype(bool)
        prot_tokens = tokens[prot_mask & standard_mask]

        neighborhood_size = random.choice(self.design_neighborhood_sizes)

        if fixed_crop:
            num_crop = min(len(prot_tokens) // 4, 20)
        else:
            num_crop = max(random.integers(len(prot_tokens)), neighborhood_size + 1)

        query = prot_tokens[random.integers(len(prot_tokens))]
        crop_indices = self.cropper.select_cropped_indices(
            tokens=prot_tokens,
            valid_tokens=prot_tokens[prot_tokens["resolved_mask"]],
            query=query,
            neighborhood_size=neighborhood_size,
            max_atoms=num_crop * 10,
            max_tokens=num_crop,
        )

        if len(crop_indices) > 0:
            design_indices = prot_tokens["token_idx"][crop_indices]
            if len(design_indices) > 0:
                tokens["design_mask"][design_indices] = True

        return tokens

    def select_motif_binder(
        self,
        tokens: np.ndarray,
        random: np.random.Generator,
        fixed_crop: bool = False,
    ):
        tokens = self.select_motif(tokens, random, fixed_crop)
        return self.resect_and_reindex(tokens, random)

    def select_scaffold(
        self,
        tokens: np.ndarray,
        random: np.random.Generator,
        fixed_crop: bool = False,
    ):
        prot_mask = tokens["mol_type"] == const.chain_type_ids["PROTEIN"]
        standard_mask = tokens["is_standard"].astype(bool)
        prot_tokens = tokens[prot_mask & standard_mask]

        neighborhood_size = random.choice(self.design_neighborhood_sizes)
        if fixed_crop:
            num_crop = min(len(prot_tokens) // 4, 20)
        else:
            num_crop = max(random.integers(len(prot_tokens)), neighborhood_size + 1)

        query = prot_tokens[random.integers(len(prot_tokens))]
        crop_indices = self.cropper.select_cropped_indices(
            tokens=prot_tokens,
            valid_tokens=prot_tokens[prot_tokens["resolved_mask"]],
            query=query,
            neighborhood_size=neighborhood_size,
            max_atoms=num_crop * 10,
            max_tokens=num_crop,
        )

        prot_tok_mask = np.ones(len(prot_tokens))
        prot_tok_mask[crop_indices] = 0

        design_indices = prot_tokens["token_idx"][prot_tok_mask.astype(bool)]

        if len(design_indices) > 0:
            tokens["design_mask"][design_indices] = True
        return tokens

    def select_scaffold_binder(
        self,
        tokens: np.ndarray,
        random: np.random.Generator,
        fixed_crop: bool = False,
    ):
        tokens = self.select_scaffold(tokens, random, fixed_crop)
        return self.resect_and_reindex(tokens, random)

    def select_standard_prot(self, tokens, random):
        prot_mask = tokens["mol_type"] == const.chain_type_ids["PROTEIN"]
        standard_mask = tokens["is_standard"].astype(bool)
        tokens["design_mask"][prot_mask & standard_mask] = True
        return tokens

    def select_nonprot_interface(self, tokens, random):
        prot_mask = tokens["mol_type"] == const.chain_type_ids["PROTEIN"]
        standard_mask = tokens["is_standard"].astype(bool)
        prot_tokens = tokens[prot_mask & standard_mask]
        nonprot_tokens = tokens[~prot_mask]

        # Get target tokens from a random number of target chains
        nonprot_chain_ids = np.unique(nonprot_tokens["asym_id"])
        num_target_chains = random.choice(np.arange(1, len(nonprot_chain_ids) + 1))
        select_ids = random.choice(nonprot_chain_ids, num_target_chains, replace=False)
        target_chains = [tokens[tokens["asym_id"] == id] for id in select_ids]
        target_tokens = np.concatenate(target_chains)

        # Get closest redesign tokens
        noisy_distances = min_token_distances(
            prot_tokens, target_tokens, random, self.distance_noise_std
        )
        indices = np.argsort(noisy_distances)
        num_selected = random.choice(np.arange(1, len(prot_tokens) + 1))
        selected_tokens = prot_tokens[indices[:num_selected]]
        tokens["design_mask"][selected_tokens["token_idx"]] = True
        return tokens

    def select_protein_chains(self, tokens, random):
        prot_mask = tokens["mol_type"] == const.chain_type_ids["PROTEIN"]
        standard_mask = tokens["is_standard"].astype(bool)
        prot_tokens = tokens[prot_mask & standard_mask]

        prot_tokens["mol_type"]
        prot_chain_ids = np.unique(prot_tokens["asym_id"])
        assert len(prot_chain_ids) > 1
        num_selections = random.choice(np.arange(1, len(prot_chain_ids)))
        select_ids = random.choice(prot_chain_ids, num_selections, replace=False)

        for id in select_ids:
            tokens["design_mask"][(id == tokens["asym_id"]) & standard_mask] = True
        return tokens

    def select_protein_intefaces(self, tokens, random):
        prot_mask = tokens["mol_type"] == const.chain_type_ids["PROTEIN"]
        standard_mask = tokens["is_standard"].astype(bool)
        prot_tokens = tokens[prot_mask & standard_mask]

        # Select chains
        prot_chain_ids = np.unique(prot_tokens["asym_id"])
        assert len(prot_chain_ids) > 1
        num_selections = random.choice(np.arange(1, len(prot_chain_ids)))
        select_ids = random.choice(prot_chain_ids, num_selections, replace=False)
        redesign_tokens = tokens[np.isin(tokens["asym_id"], select_ids) & standard_mask]
        target_tokens = tokens[~np.isin(tokens["asym_id"], select_ids)]

        # Get indices of closest redesign tokens to the target tokens
        noisy_distances = min_token_distances(
            redesign_tokens, target_tokens, random, self.distance_noise_std
        )
        indices = np.argsort(noisy_distances)
        num_selected = random.choice(np.arange(1, len(redesign_tokens) + 1))
        selected_tokens = redesign_tokens[indices[:num_selected]]
        tokens["design_mask"][selected_tokens["token_idx"]] = True
        return tokens
