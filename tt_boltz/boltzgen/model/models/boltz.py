"""Boltz model — vendored from BoltzGen with PyTorch-Lightning removed.

The class is now a plain ``nn.Module``. Training-only methods
(``training_step``, ``validation_step``, ``configure_optimizers``,
``on_*_epoch_end``, ``on_before_optimizer_step``, ``training_log``,
``configure_callbacks``, ``on_load_checkpoint``) are gone, along with
their associated loss / optimizer / validator imports and the
``MeanMetric`` training-loss loggers. Training-time constructor args are
still accepted for backward compatibility with older checkpoints but
their values are ignored.

``convert_to_tt`` is applied at the end of ``__init__`` so every
constructed model runs on Tenstorrent — there's no CPU/GPU path here.
"""

import math
from pathlib import Path
import time
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from torch import Tensor, nn

from tt_boltz.tenstorrent import (
    MiniformerModule as TTMiniformerModule,
    MSAModule as TTMSAModule,
    PairformerModule as TTPairformerModule,
)

from tt_boltz.boltzgen.data.rmsd_computation import get_true_coordinates
import tt_boltz.boltzgen.model.layers.initialize as init
from tt_boltz.data import const
from tt_boltz.data.mol import minimum_lddt_symmetry_dist
from tt_boltz.boltzgen.model.modules.confidence import ConfidenceModule
from tt_boltz.boltzgen.model.modules.diffusion import AtomDiffusion
from tt_boltz.boltzgen.model.modules.diffusion_conditioning import (
    DiffusionConditioning,
)
from tt_boltz.boltzgen.model.modules.encoders import RelativePositionEncoder
from tt_boltz.boltzgen.model.modules.affinity import AffinityModule
from tt_boltz.boltzgen.model.modules.masker import BoltzMasker
from tt_boltz.boltzgen.model.modules.trunk import (
    BFactorModule,
    ContactConditioning,
    DistogramModule,
    InputEmbedder,
    TemplateModule,
    TokenDistanceModule,
)
from tt_boltz.boltzgen.model.modules.inverse_fold import (
    InverseFoldingEncoder,
    InverseFoldingDecoder,
)


class Boltz(nn.Module):
    """Boltz model (vendored from BoltzGen, inference-only)."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: Dict[str, Any],
        validation_args: Dict[str, Any],
        embedder_args: Dict[str, Any],
        msa_args: Dict[str, Any],
        pairformer_args: Dict[str, Any],
        score_model_args: Dict[str, Any],
        diffusion_process_args: Dict[str, Any],
        diffusion_loss_args: Dict[str, Any],
        affinity_model_args: Dict[str, Any] = {},
        affinity_mw_correction: bool = True,
        affinity_ensemble: bool = False,
        affinity_model_args1: Dict[str, Any] = {},
        affinity_model_args2: Dict[str, Any] = {},
        confidence_model_args: Optional[Dict[str, Any]] = None,
        validators: Any = None,
        masker_args: dict[str, Any] = {},
        num_val_datasets: int = 1,
        atom_feature_dim: int = 128,
        template_args: Optional[Dict] = None,
        use_miniformer: bool = True,
        confidence_prediction: bool = False,
        affinity_prediction: bool = False,
        token_level_confidence: bool = True,
        alpha_pae: float = 0.0,
        structure_prediction_training: bool = True,
        validate_structure: bool = True,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        exclude_ions_from_lddt: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        ignore_ckpt_shape_mismatch: bool = False,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        predict_args: Optional[Dict[str, Any]] = None,
        checkpoints: Optional[Dict[str, Any]] = None,
        step_scale_schedule: Optional[List[Dict[str, float]]] = None,
        noise_scale_schedule: Optional[List[Dict[str, float]]] = None,
        aggregate_distogram: bool = True,
        bond_type_feature: bool = False,
        no_random_recycling_training: bool = False,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        use_templates: bool = False,
        use_token_distances: bool = False,
        token_distance_args: Optional[Dict] = None,
        predict_bfactor: bool = False,
        log_loss_every_steps: int = 50,
        checkpoint_diffusion_conditioning: bool = False,
        freeze_template_weights: bool = False,
        refolding_validator=None,
        predict_res_type: bool = False,
        inverse_fold: bool = False,
        inverse_fold_args: Optional[Dict[str, Any]] = None,
        inference_logging: bool = False,
        use_kernels: bool = False,
    ) -> None:
        super().__init__()
        """
        Module that does either:
        1. Design
        2. Folding with confidence prediction
        3. Inverse folding
        4. Affinity prediction
        """
        self.inverse_fold = inverse_fold
        self.inference_logging = inference_logging
        self.use_kernels = use_kernels
        self.ignore_ckpt_shape_mismatch = ignore_ckpt_shape_mismatch

        # Inference-relevant args
        self.predict_args = predict_args
        self.predict_res_type = predict_res_type

        # Checkpoints (multi-ckpt schedule used at inference time)
        self.checkpoints = checkpoints
        self.inference_counter = 0
        if checkpoints:
            self.first_checkpoint_num_samples = checkpoints.get(
                "first_checkpoint_num_samples", 1.0
            )
            self.checkpoint_list = checkpoints.get("checkpoint_list", [])
            self.total_samples = None
            self.switch_points = []
            self.checkpoint_paths = []
            self.current_checkpoint_index = -1

        # Noise and step scales (used at inference time per predict_step)
        self.step_scale_schedule = step_scale_schedule
        self.noise_scale_schedule = noise_scale_schedule
        self.step_scale_switch_points = []
        self.step_scale_values = []
        self.noise_scale_switch_points = []
        self.noise_scale_values = []

        if "affinity_args" not in affinity_model_args:
            affinity_model_args["affinity_args"] = {}
        if "groups" not in affinity_model_args["affinity_args"]:
            affinity_model_args["affinity_args"]["groups"] = {0: 1}
        if "val_groups" not in affinity_model_args["affinity_args"]:
            affinity_model_args["affinity_args"]["val_groups"] = set([0])

        self.exclude_ions_from_lddt = exclude_ions_from_lddt

        # Distogram
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.aggregate_distogram = aggregate_distogram

        # Trunk
        self.use_miniformer = use_miniformer

        # Masker
        self.masker = BoltzMasker(**masker_args)

        # Input embeddings
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            **embedder_args,
        }
        if not self.inverse_fold:
            self.input_embedder = InputEmbedder(**full_embedder_args)

            self.s_init = nn.Linear(token_s, token_s, bias=False)
            self.z_init_1 = nn.Linear(token_s, token_z, bias=False)
            self.z_init_2 = nn.Linear(token_s, token_z, bias=False)

            self.rel_pos = RelativePositionEncoder(token_z)

            self.token_bonds = nn.Linear(
                1,
                token_z,
                bias=False,
            )
            self.bond_type_feature = bond_type_feature
            if bond_type_feature:
                self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

            self.contact_conditioning = ContactConditioning(
                token_z=token_z,
                cutoff_min=conditioning_cutoff_min,
                cutoff_max=conditioning_cutoff_max,
            )

            # Normalization layers
            self.s_norm = nn.LayerNorm(token_s)
            self.z_norm = nn.LayerNorm(token_z)

            # Recycling projections
            self.s_recycle = nn.Linear(token_s, token_s, bias=False)
            self.z_recycle = nn.Linear(token_z, token_z, bias=False)
            init.gating_init_(self.s_recycle.weight)
            init.gating_init_(self.z_recycle.weight)

        # Pairwise stack
        self.use_token_distances = use_token_distances
        if self.use_token_distances:
            self.token_distance_module = TokenDistanceModule(
                token_z, **token_distance_args
            )

        self.use_templates = use_templates

        if use_templates:
            self.template_module = TemplateModule(token_z, **template_args)

        if not self.inverse_fold:
            # Direct ttnn construction — no PyTorch fallback path. MSA + Pairformer
            # are the heavy modules and run on the TT device; their state_dict
            # layout is byte-identical to BoltzGen's PyTorch versions, so the
            # production checkpoints load cleanly via the TorchWrapper.
            self.msa_module = TTMSAModule(
                n_blocks=msa_args["msa_blocks"],
                avg_head_dim=32, avg_n_heads=8,
                tri_att_head_dim=32, tri_att_n_heads=4,
            )
            num_heads = pairformer_args.get("num_heads", 16)
            att_head_dim = token_s // num_heads
            if use_miniformer:
                self.pairformer_module = TTMiniformerModule(
                    n_blocks=pairformer_args["num_blocks"],
                    att_head_dim=att_head_dim, att_n_heads=num_heads,
                )
            else:
                self.pairformer_module = TTPairformerModule(
                    n_blocks=pairformer_args["num_blocks"],
                    tri_att_head_dim=pairformer_args.get("pairwise_head_width", 32),
                    tri_att_n_heads=pairformer_args.get("pairwise_num_heads", 4),
                    att_head_dim=att_head_dim, att_n_heads=num_heads,
                    transform_s=True,
                )
            self.checkpoint_diffusion_conditioning = checkpoint_diffusion_conditioning

            self.diffusion_conditioning = DiffusionConditioning(
                token_s=token_s,
                token_z=token_z,
                atom_s=atom_s,
                atom_z=atom_z,
                atoms_per_window_queries=atoms_per_window_queries,
                atoms_per_window_keys=atoms_per_window_keys,
                atom_encoder_depth=score_model_args["atom_encoder_depth"],
                atom_encoder_heads=score_model_args["atom_encoder_heads"],
                token_transformer_depth=score_model_args["token_transformer_depth"],
                token_transformer_heads=score_model_args["token_transformer_heads"],
                atom_decoder_depth=score_model_args["atom_decoder_depth"],
                atom_decoder_heads=score_model_args["atom_decoder_heads"],
                atom_feature_dim=atom_feature_dim,
                conditioning_transition_layers=score_model_args[
                    "conditioning_transition_layers"
                ],
            )

            # Output modules
            self.structure_module = AtomDiffusion(
                score_model_args={
                    "token_s": token_s,
                    "atom_s": atom_s,
                    "atoms_per_window_queries": atoms_per_window_queries,
                    "atoms_per_window_keys": atoms_per_window_keys,
                    "predict_res_type": predict_res_type,
                    **score_model_args,
                },
                **diffusion_process_args,
            )
            self.distogram_module = DistogramModule(token_z, num_bins)
            self.predict_bfactor = predict_bfactor
            if predict_bfactor:
                self.bfactor_module = BFactorModule(token_s, num_bins)

        self.confidence_prediction = confidence_prediction
        self.token_level_confidence = token_level_confidence
        self.alpha_pae = alpha_pae

        self.structure_prediction_training = structure_prediction_training

        ### Affinity ###
        self.affinity_prediction = affinity_prediction
        self.affinity_ensemble = affinity_ensemble
        self.affinity_mw_correction = affinity_mw_correction

        if self.affinity_prediction:
            if self.affinity_ensemble:
                self.affinity_module1 = AffinityModule(
                    token_s,
                    token_z,
                    **affinity_model_args1,
                )
                self.affinity_module2 = AffinityModule(
                    token_s,
                    token_z,
                    **affinity_model_args2,
                )
            else:
                self.affinity_module = AffinityModule(
                    token_s,
                    token_z,
                    **affinity_model_args,
                )

        if self.confidence_prediction:
            self.confidence_module = ConfidenceModule(
                token_s,
                token_z,
                token_level_confidence=token_level_confidence,
                bond_type_feature=bond_type_feature,
                conditioning_cutoff_min=conditioning_cutoff_min,
                conditioning_cutoff_max=conditioning_cutoff_max,
                **confidence_model_args,
            )

        if self.inverse_fold:
            self.enable_if_input_embedder = False
            if inverse_fold_args.get("enable_input_embedder", False):
                self.enable_if_input_embedder = True
                self.input_embedder = InputEmbedder(**full_embedder_args)
            self.predict_bfactor = False
            self.inverse_folding_encoder = InverseFoldingEncoder(**inverse_fold_args)
            self.structure_module = InverseFoldingDecoder(**inverse_fold_args)

    def setup_for_inference(self, dataloader) -> None:
        """Compute per-run inference schedules from the loaded dataloader.

        Replaces ``LightningModule.setup`` — the validator/training half of
        that hook is gone (we never call .setup() with stage != "predict"),
        and the predict half is rewritten to take the dataloader as a plain
        argument instead of reading ``self.trainer.datamodule``.
        """
        self.total_samples = len(dataloader.dataset)

        if self.checkpoints:
            fractions = [self.first_checkpoint_num_samples] + [
                ckpt["checkpoint"]["num_samples"] for ckpt in self.checkpoint_list
            ]
            cumulative_samples = np.cumsum(
                [int(math.ceil(f * self.total_samples)) for f in fractions]
            )
            self.switch_points = cumulative_samples.tolist()
            self.checkpoint_paths = [
                ckpt["checkpoint"]["path"] for ckpt in self.checkpoint_list
            ]
            for path in self.checkpoint_paths:
                if not Path(path).exists():
                    raise ValueError(f"Missing checkpoint path: {path}")
            self.next_switch_point = (
                self.switch_points[0] if self.switch_points else None
            )

        if self.step_scale_schedule:
            self.step_scale_values = [
                item["step_scale"] for item in self.step_scale_schedule
            ]
            fractions = [item["period"] for item in self.step_scale_schedule]
            cumulative_samples = np.cumsum(
                [int(f * self.total_samples) for f in fractions]
            )
            self.step_scale_switch_points = cumulative_samples.tolist()
            self.current_step_scale_index = 0
            self.next_step_scale_switch_point = self.step_scale_switch_points[0]
            self.current_step_scale = self.step_scale_values[0]

        if self.noise_scale_schedule:
            self.noise_scale_values = [
                item["noise_scale"] for item in self.noise_scale_schedule
            ]
            fractions = [item["period"] for item in self.noise_scale_schedule]
            cumulative_samples = np.cumsum(
                [int(f * self.total_samples) for f in fractions]
            )
            self.noise_scale_switch_points = cumulative_samples.tolist()
            self.current_noise_scale_index = 0
            self.next_noise_scale_switch_point = self.noise_scale_switch_points[0]
            self.current_noise_scale = self.noise_scale_values[0]

    @property
    def device(self) -> torch.device:
        """Tenstorrent-only — host scaffolding around ttnn runs on CPU."""
        return torch.device("cpu")

    def load_checkpoint_weights(self, checkpoint_path):
        """Hot-swap weights mid-pipeline (used by the multi-ckpt schedule)."""
        from tt_boltz.boltzgen.adapter import (
            _legacy_pickle_compat,
            _remap_legacy_state_dict_keys,
        )
        # Same legacy-pickle shim load_boltz_checkpoint() uses: shipping
        # checkpoints pickle hyper_parameters referencing deleted training-only
        # classes (boltzgen.model.validation.*), so torch.load needs the finder.
        with _legacy_pickle_compat():
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        state = _remap_legacy_state_dict_keys(checkpoint["state_dict"])
        self.load_state_dict(state, strict=False)
        print(f"Loaded weights from {checkpoint_path}")

    def forward(
        self,
        feats: Dict[str, Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        run_confidence_sequentially: bool = False,
        return_z_feats: bool = False,
        step_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        dict_out = {}
        if self.inference_logging:
            print("\nRunning Trunk.\n")
        with torch.set_grad_enabled(
            (self.training and self.structure_prediction_training)
        ):
            if self.inverse_fold:
                if self.enable_if_input_embedder:
                    s_inputs = self.input_embedder(feats)
                    feats["s_inputs"] = s_inputs
                edge_idx, valid_mask, s, z = self.inverse_folding_encoder(feats)
                # Remove s_inputs from feats dictionary
                feats.pop("s_inputs", None)
            else:
                s_inputs = self.input_embedder(feats)

                # Initialize the sequence embeddings
                s_init = self.s_init(s_inputs)

                # Initialize pairwise embeddings
                z_init = (
                    self.z_init_1(s_inputs)[:, :, None]
                    + self.z_init_2(s_inputs)[:, None, :]
                )
                relative_position_encoding = self.rel_pos(feats)
                z_init = z_init + relative_position_encoding
                z_init = z_init + self.token_bonds(feats["token_bonds"].float())
                if self.bond_type_feature:
                    z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
                z_init = z_init + self.contact_conditioning(feats)

                # Perform rounds of the pairwise stack
                s = torch.zeros_like(s_init)
                z = torch.zeros_like(z_init)

                # Compute pairwise mask
                mask = feats["token_pad_mask"].float()
                pair_mask = mask[:, :, None] * mask[:, None, :]

            if not self.inverse_fold:
                for i in range(recycling_steps + 1):
                    from tt_boltz.boltzgen.progress import progress as _emit_progress
                    _emit_progress("trunk", i + 1, recycling_steps + 1)
                    with torch.set_grad_enabled(
                        (
                            self.training
                            and self.structure_prediction_training
                            and (i == recycling_steps)
                        )
                    ):
                        # Issue with unused parameters in autocast
                        if (
                            self.training
                            and (i == recycling_steps)
                            and torch.is_autocast_enabled()
                        ):
                            torch.clear_autocast_cache()

                        # Apply recycling
                        s = s_init + self.s_recycle(self.s_norm(s))
                        z = z_init + self.z_recycle(self.z_norm(z))

                        # Compute pairwise stack
                        if self.use_token_distances:
                            z = z + self.token_distance_module(
                                z, feats, pair_mask, relative_position_encoding
                            )

                        # Compute pairwise stack
                        if self.use_templates:
                            z = z + self.template_module(
                                z, feats, pair_mask, use_kernels=self.use_kernels
                            )

                        if not self.inverse_fold:
                            z = z + self.msa_module(
                                z, s_inputs, feats, use_kernels=self.use_kernels
                            )

                        s, z = self.pairformer_module(
                            s,
                            z,
                            mask=mask,
                            pair_mask=pair_mask,
                            use_kernels=self.use_kernels,
                        )

            if not self.inverse_fold:
                pdistogram = self.distogram_module(z)
                dict_out["pdistogram"] = pdistogram.float()

            if not self.inverse_fold:
                if self.checkpoint_diffusion_conditioning:
                    # TODO decide whether this should be with bf16 or not
                    (
                        q,
                        c,
                        to_keys,
                        atom_enc_bias,
                        atom_dec_bias,
                        token_trans_bias,
                    ) = torch.utils.checkpoint.checkpoint(
                        self.diffusion_conditioning,
                        s,
                        z,
                        relative_position_encoding,
                        feats,
                    )
                else:
                    (
                        q,
                        c,
                        to_keys,
                        atom_enc_bias,
                        atom_dec_bias,
                        token_trans_bias,
                    ) = self.diffusion_conditioning(
                        s_trunk=s,
                        z_trunk=z,
                        relative_position_encoding=relative_position_encoding,
                        feats=feats,
                    )
                diffusion_conditioning = {
                    "q": q,
                    "c": c,
                    "to_keys": to_keys,
                    "atom_enc_bias": atom_enc_bias,
                    "atom_dec_bias": atom_dec_bias,
                    "token_trans_bias": token_trans_bias,
                }

                if self.predict_bfactor:
                    pbfactor = self.bfactor_module(s)
                    dict_out["pbfactor"] = pbfactor

            if (
                (not self.training)
                or self.confidence_prediction
                or self.affinity_prediction
            ):
                if self.inference_logging:
                    print("\nRunning Structure Module.\n")
                with torch.autocast("cuda", enabled=False):
                    if not self.inverse_fold:
                        struct_out = self.structure_module.sample(
                            s_trunk=s.float(),
                            s_inputs=s_inputs.float(),
                            feats=feats,
                            num_sampling_steps=num_sampling_steps,
                            atom_mask=feats["atom_pad_mask"].float(),
                            multiplicity=diffusion_samples,
                            diffusion_conditioning=diffusion_conditioning,
                            step_scale=step_scale,
                            noise_scale=noise_scale,
                            inference_logging=self.inference_logging,
                        )
                    else:
                        struct_out = self.structure_module.sample(
                            s=s,
                            z=z,
                            edge_idx=edge_idx,
                            valid_mask=valid_mask,
                            feats=feats,
                        )

                    dict_out.update(struct_out)

                if self.training and self.structure_prediction_training:
                    for idx in range(feats["token_index"].shape[0]):
                        minimum_lddt_symmetry_dist(
                            pred_distogram=pdistogram[idx],
                            feats=feats,
                            index_batch=idx,
                        )

            if self.training and (
                self.confidence_prediction or self.affinity_prediction
            ):
                assert len(feats["coords"].shape) == 4
                assert feats["coords"].shape[1] == 1, (
                    "Only one conformation is supported for confidence"
                )

            # Compute structure module
            if self.training and self.structure_prediction_training:
                atom_coords = feats["coords"]
                B, K, L = atom_coords.shape[0:3]
                assert K in (
                    multiplicity_diffusion_train,
                    1,
                )  # TODO make check somewhere else, expand to m % N == 0, m > N
                atom_coords = atom_coords.reshape(B * K, L, 3)
                atom_coords = atom_coords.repeat_interleave(
                    multiplicity_diffusion_train // K, 0
                )
                feats["coords"] = atom_coords  # (multiplicity, L, 3)
                assert len(feats["coords"].shape) == 3

                with torch.autocast("cuda", enabled=False):
                    if not self.inverse_fold:
                        struct_out = self.structure_module(
                            s_trunk=s.float(),
                            s_inputs=s_inputs.float(),
                            feats=feats,
                            multiplicity=multiplicity_diffusion_train,
                            diffusion_conditioning=diffusion_conditioning,
                        )
                    else:
                        struct_out = self.structure_module(
                            s=s,
                            z=z,
                            edge_idx=edge_idx,
                            valid_mask=valid_mask,
                            feats=feats,
                        )
                    dict_out.update(struct_out)

            elif self.training:
                feats["coords"] = feats["coords"].squeeze(1)
                assert len(feats["coords"].shape) == 3

        if self.confidence_prediction:
            dict_out.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    x_pred=(dict_out["sample_atom_coords"].detach()),
                    feats=feats,
                    pred_distogram_logits=(dict_out["pdistogram"][:, :, :, 0].detach()),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
            )

        if self.affinity_prediction:
            pad_token_mask = feats["token_pad_mask"][0]
            rec_mask = feats["mol_type"][0] == 0
            rec_mask = rec_mask * pad_token_mask
            lig_mask = feats["affinity_token_mask"][0].to(torch.bool)
            lig_mask = lig_mask * pad_token_mask
            cross_pair_mask = (
                lig_mask[:, None] * rec_mask[None, :]
                + rec_mask[:, None] * lig_mask[None, :]
                + lig_mask[:, None] * lig_mask[None, :]
            )
            z_affinity = z * cross_pair_mask[None, :, :, None]

            argsort = torch.argsort(dict_out["iptm"], descending=True)
            best_idx = argsort[0].item()
            coords_affinity = dict_out["sample_atom_coords"].detach()[best_idx][
                None, None
            ]
            s_inputs = self.input_embedder(feats, affinity=True)

            with torch.autocast("cuda", enabled=False):
                if self.affinity_ensemble:
                    dict_out_affinity1 = self.affinity_module1(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )

                    dict_out_affinity1["affinity_probability_binary"] = (
                        torch.nn.functional.sigmoid(
                            dict_out_affinity1["affinity_logits_binary"]
                        )
                    )
                    dict_out_affinity2 = self.affinity_module2(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )
                    dict_out_affinity2["affinity_probability_binary"] = (
                        torch.nn.functional.sigmoid(
                            dict_out_affinity2["affinity_logits_binary"]
                        )
                    )

                    dict_out_affinity_ensemble = {
                        "affinity_pred_value": (
                            dict_out_affinity1["affinity_pred_value"]
                            + dict_out_affinity2["affinity_pred_value"]
                        )
                        / 2,
                        "affinity_probability_binary": (
                            dict_out_affinity1["affinity_probability_binary"]
                            + dict_out_affinity2["affinity_probability_binary"]
                        )
                        / 2,
                    }

                    dict_out_affinity1 = {
                        "affinity_pred_value1": dict_out_affinity1[
                            "affinity_pred_value"
                        ],
                        "affinity_probability_binary1": dict_out_affinity1[
                            "affinity_probability_binary"
                        ],
                    }
                    dict_out_affinity2 = {
                        "affinity_pred_value2": dict_out_affinity2[
                            "affinity_pred_value"
                        ],
                        "affinity_probability_binary2": dict_out_affinity2[
                            "affinity_probability_binary"
                        ],
                    }

                    if self.affinity_mw_correction:
                        model_coef = 1.03525938
                        mw_coef = -0.59992683
                        bias = 2.83288489
                        mw = feats["affinity_mw"][0] ** 0.3
                        dict_out_affinity_ensemble["affinity_pred_value"] = (
                            model_coef
                            * dict_out_affinity_ensemble["affinity_pred_value"]
                            + mw_coef * mw
                            + bias
                        )

                    dict_out.update(dict_out_affinity_ensemble)
                    dict_out.update(dict_out_affinity1)
                    dict_out.update(dict_out_affinity2)
                else:
                    dict_out_affinity = self.affinity_module(
                        s_inputs=s_inputs.detach(),
                        z=z_affinity.detach(),
                        x_pred=coords_affinity,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=self.use_kernels,
                    )
                    dict_out.update(
                        {
                            "affinity_pred_value": dict_out_affinity[
                                "affinity_pred_value"
                            ],
                            "affinity_probability_binary": torch.nn.functional.sigmoid(
                                dict_out_affinity["affinity_logits_binary"]
                            ),
                        }
                    )
        if return_z_feats:
            dict_out["z_feats"] = z

        # For stability checking as in, https://github.com/IntelliGen-AI/IntFold
        dict_out["s_trunk"] = s
        return dict_out

    def predict_step(
        self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Dict:
        # Skip invalid samples
        if "exception" in batch and any(batch["exception"]):
            print(f"WARNING: Skipping batch. Exception for {batch['id'][0]}")
            return {"exception": True}
        if "skip" in batch and any(batch["skip"]):
            print(f"WARNING: Skipping batch. Skip was set true for {batch['id'][0]}")
            return {"skip": True}

        # Tenstorrent: the ttnn modules cache per-design static data (masks,
        # biases, q/c projections, atom padding) so it is computed once and
        # reused across the diffusion sampling steps. That cache is only valid
        # for a single design — reset it for each new design. Otherwise a design
        # with a different atom/token count reuses the previous design's cached
        # atom padding, mismatching tensor shapes and crashing ttnn.add with
        # "Invalid subtile broadcast type". Mirrors the main path (boltz2.py).
        for m in self.modules():
            if hasattr(m, "reset_static_cache"):
                m.reset_static_cache()

        # Checkpoint switching logic
        if self.checkpoints and self.next_switch_point is not None:
            if self.inference_counter == self.next_switch_point:
                self.current_checkpoint_index += 1
                if 0 <= self.current_checkpoint_index < len(self.checkpoint_paths):
                    checkpoint_path = self.checkpoint_paths[
                        self.current_checkpoint_index
                    ]
                    self.load_checkpoint_weights(checkpoint_path)
                    print(f"Switched checkpoint.")
                if self.current_checkpoint_index + 1 < len(self.switch_points):
                    self.next_switch_point = self.switch_points[
                        self.current_checkpoint_index + 1
                    ]
                else:
                    self.next_switch_point = None

        # Temperature switching logic
        if self.step_scale_schedule and self.next_step_scale_switch_point is not None:
            if self.inference_counter == self.next_step_scale_switch_point:
                self.current_step_scale_index += 1
                if self.current_step_scale_index < len(self.step_scale_values):
                    self.current_step_scale = self.step_scale_values[
                        self.current_step_scale_index
                    ]
                    if self.current_step_scale_index + 1 < len(
                        self.step_scale_switch_points
                    ):
                        self.next_step_scale_switch_point = (
                            self.step_scale_switch_points[
                                self.current_step_scale_index + 1
                            ]
                        )
                    else:
                        self.next_step_scale_switch_point = None
                print(f"Switched step_scale to {self.current_step_scale}")

        if self.noise_scale_schedule and self.next_noise_scale_switch_point is not None:
            if self.inference_counter == self.next_noise_scale_switch_point:
                self.current_noise_scale_index += 1
                if self.current_noise_scale_index < len(self.noise_scale_values):
                    self.current_noise_scale = self.noise_scale_values[
                        self.current_noise_scale_index
                    ]
                    if self.current_noise_scale_index + 1 < len(
                        self.noise_scale_switch_points
                    ):
                        self.next_noise_scale_switch_point = (
                            self.noise_scale_switch_points[
                                self.current_noise_scale_index + 1
                            ]
                        )
                    else:
                        self.next_noise_scale_switch_point = None
                print(f"Switched noise_scale to {self.current_noise_scale}")

        step_scale = getattr(self, "current_step_scale", None)
        noise_scale = getattr(self, "current_noise_scale", None)

        try:
            feat_masked = self.masker(batch)
            out = self(
                feat_masked,
                recycling_steps=self.predict_args["recycling_steps"],
                num_sampling_steps=self.predict_args["sampling_steps"],
                diffusion_samples=self.predict_args["diffusion_samples"],
                run_confidence_sequentially=True,
                step_scale=step_scale,
                noise_scale=noise_scale,
                return_z_feats=(
                    self.predict_args["return_z_feats"]
                    if "return_z_feats" in self.predict_args
                    else False
                ),
            )
            pred_dict = {"exception": False}
            pred_dict.update(feat_masked)

            if "keys_dict_batch" in self.predict_args:
                for key in self.predict_args["keys_dict_batch"]:
                    pred_dict[key] = batch[key]
            if (
                "return_z_feats" in self.predict_args
                and self.predict_args["return_z_feats"]
            ):
                pred_dict["z_feats"] = out["z_feats"]
            pred_dict["masks"] = batch["atom_pad_mask"]
            pred_dict["token_masks"] = batch["token_pad_mask"]
            if "keys_dict_out" in self.predict_args:
                for key in self.predict_args["keys_dict_out"]:
                    pred_dict[key] = out[key]

            # also save these keys for computing refolding metrics like scRMSD
            pred_dict["input_coords"] = batch["coords"]
            pred_dict["token_index"] = batch["token_index"]
            pred_dict["atom_resolved_mask"] = batch["atom_resolved_mask"]
            pred_dict["atom_to_token"] = batch["atom_to_token"]
            pred_dict["mol_type"] = batch["mol_type"]
            pred_dict["backbone_mask"] = batch["backbone_mask"]

            pred_dict["coords"] = out["sample_atom_coords"]
            if not self.inverse_fold:
                pred_dict["coords_traj"] = out["coords_traj"]
                pred_dict["x0_coords_traj"] = out["x0_coords_traj"]
            if self.confidence_prediction:
                # pred_dict["confidence"] = out.get("ablation_confidence", None)
                pred_dict["pde"] = out["pde"]
                pred_dict["plddt"] = out["plddt"]
                pred_dict["confidence_score"] = (
                    4 * out["complex_plddt"]
                    + (
                        out["iptm"]
                        if not torch.allclose(
                            out["iptm"], torch.zeros_like(out["iptm"])
                        )
                        else out["ptm"]
                    )
                ) / 5

                pred_dict["complex_plddt"] = out["complex_plddt"]
                pred_dict["complex_iplddt"] = out["complex_iplddt"]
                pred_dict["complex_pde"] = out["complex_pde"]
                pred_dict["complex_ipde"] = out["complex_ipde"]
                if self.alpha_pae > 0:
                    pred_dict["pae"] = out["pae"]
                    pred_dict["ptm"] = out["ptm"]
                    pred_dict["iptm"] = out["iptm"]
                    pred_dict["ligand_iptm"] = out["ligand_iptm"]
                    pred_dict["protein_iptm"] = out["protein_iptm"]
                    pred_dict["pair_chains_iptm"] = out["pair_chains_iptm"]
                    pred_dict["design_ipsae_min"] = out["design_ipsae_min"]
                    pred_dict["design_to_target_ipsae"] = out["design_to_target_ipsae"]
                    pred_dict["target_to_design_ipsae"] = out["target_to_design_ipsae"]

                if self.affinity_prediction:
                    pred_dict["affinity_pred_value"] = out["affinity_pred_value"]
                    pred_dict["affinity_probability_binary"] = out[
                        "affinity_probability_binary"
                    ]
                    if self.affinity_ensemble:
                        pred_dict["affinity_pred_value1"] = out["affinity_pred_value1"]
                        pred_dict["affinity_probability_binary1"] = out[
                            "affinity_probability_binary1"
                        ]
                        pred_dict["affinity_pred_value2"] = out["affinity_pred_value2"]
                        pred_dict["affinity_probability_binary2"] = out[
                            "affinity_probability_binary2"
                        ]
            self.inference_counter += 1
            return pred_dict

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                return {"exception": True}
            else:
                raise e

