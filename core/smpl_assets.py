"""SMPL asset contract checks used by NLF fitting."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .motionpack.io import _file_sha256
from .motionpack.model import (
    BODY_MODEL_ASSET_IDENTITY,
    SMPL_VERTEX_SUBSET_SHA256,
    validate_body_model_asset_identity,
)

EXPECTED_SMPL_VERTEX_SUBSET_SIZE = 1024
VERTEX_SUBSET_FILE = "vertex_subset_1024.npz"
JOINT_REGRESSOR_FILE = "vertex_subset_joint_regr_post_lbs_1024.npy"


def current_body_model_asset_identity() -> dict:
    import config as cfg

    files = {
        "model_sha256": cfg.SMPL_NEUTRAL_PATH,
        "kid_template_sha256": cfg.SMPL_FITTER_ROOT / "kid_template.npy",
        "joint_regressor_sha256": cfg.SMPL_FITTER_ROOT / JOINT_REGRESSOR_FILE,
        "faces_sha256": cfg.SMPL_FACES_6890_PATH,
    }
    identity = {**BODY_MODEL_ASSET_IDENTITY, **{field: _file_sha256(path) for field, path in files.items()}}
    return validate_body_model_asset_identity(identity)


def smpl_vertex_subset_contract_status(model_root: str | Path) -> dict:
    path = Path(model_root) / VERTEX_SUBSET_FILE
    status = {
        "path": str(path),
        "exists": path.exists(),
        "expected_size": EXPECTED_SMPL_VERTEX_SUBSET_SIZE,
        "expected_sha256": SMPL_VERTEX_SUBSET_SHA256,
        "size": 0,
        "sha256": "",
        "is_valid": False,
    }
    if not status["exists"]:
        return status
    with np.load(path) as data:
        subset = np.asarray(data["i_verts"], dtype="<i8")
    status.update(size=len(subset), sha256=hashlib.sha256(subset.tobytes()).hexdigest())
    status["is_valid"] = (
        status["size"] == EXPECTED_SMPL_VERTEX_SUBSET_SIZE
        and status["sha256"] == SMPL_VERTEX_SUBSET_SHA256
    )
    return status
