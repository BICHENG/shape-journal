"""Fit the fixed SMPL model to the 1048 camera-space points in a PosePack."""

import torch

import config as cfg
from .motionpack.model import SMPL_24_JOINT_SCHEMA, validate_joint_schema
from .smpl_assets import current_body_model_asset_identity, smpl_vertex_subset_contract_status


FIT_OPTIONS = dict(num_iter=3, beta_regularizer=10.0, beta_regularizer2=0.0, final_adjust_rots=True)
UNCERTAINTY_WEIGHT_POWER = -1.5


class SMPLFitter:
    def __init__(self, device: str):
        from smplfitter.pt import BodyFitter, BodyModel

        self.asset_identity = current_body_model_asset_identity()
        subset = smpl_vertex_subset_contract_status(cfg.SMPL_FITTER_ROOT)
        if not subset["is_valid"]:
            raise ValueError(f"SMPL 1024 vertex subset does not match the 1048 query layout: {subset['path']}")
        self.subset_sha256 = subset["sha256"]
        root = str(cfg.SMPL_FITTER_ROOT)
        partial = BodyModel("smpl", num_betas=10, model_root=root, vertex_subset_size=1024)
        self.vertex_count = partial.num_vertices
        self.fitter = BodyFitter(partial).to(device)
        self.model = BodyModel("smpl", num_betas=10, model_root=root).to(device)
        self.joint_schema = validate_joint_schema(dict(
            SMPL_24_JOINT_SCHEMA,
            names=tuple(self.model.joint_names),
            edges=tuple(
                (int(parent), index)
                for index, parent in enumerate(self.model.kintree_parents) if parent >= 0
            ),
        ))

    @torch.inference_mode()
    def __call__(self, points_mm, uncertainty_mm):
        center = points_mm.mean(dim=-2, keepdim=True)
        points = (points_mm - center) / 1000
        vertex_weights, joint_weights = uncertainty_mm.pow(UNCERTAINTY_WEIGHT_POWER).split(
            (self.vertex_count, self.model.num_joints), dim=-1,
        )
        fit = self.fitter.fit(
            points[:, :self.vertex_count], points[:, self.vertex_count:],
            vertex_weights=vertex_weights / vertex_weights.mean(dim=-1, keepdim=True),
            joint_weights=joint_weights / joint_weights.mean(dim=-1, keepdim=True),
            requested_keys=["pose_rotvecs", "shape_betas", "trans"],
            **FIT_OPTIONS,
        )
        body = self.model(fit["pose_rotvecs"], fit["shape_betas"], fit["trans"], return_vertices=False)
        return {
            "betas": fit["shape_betas"],
            "pose_rotvecs": fit["pose_rotvecs"],
            "joints3d": body["joints"] * 1000 + center,
        }
