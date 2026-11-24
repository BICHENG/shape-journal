"""Runtime paths for model assets and compiled TensorRT engines."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("NLF_MODEL_ROOT", PROJECT_ROOT / "models")).resolve()
NLF_WEIGHTS_ROOT = PROJECT_ROOT / "weights" / "main"
NLF_WEIGHT_FILES = ("backbone.pth", "backbone_buffers.pth", "buffers.pth", "field.pth", "head.pth")
NLF_CANONICAL_PATH = PROJECT_ROOT / "output" / "canonical_smpl.npy"
NLF_METADATA_SNAPSHOT_PATH = PROJECT_ROOT / "body_measurement" / "specs" / "nlf_metadata_snapshot.json"
YOLO_MODEL_DIR = MODEL_ROOT / "detector" / "yolo26"
SMPL_MODEL_ROOT = Path(os.environ.get("SMPL_MODEL_ROOT", MODEL_ROOT / "smpl_models")).resolve()
SMPL_FITTER_ROOT = Path(os.environ.get("SMPL_FITTER_ROOT", SMPL_MODEL_ROOT / "smpl")).resolve()
TRT_CACHE_ROOT = Path(os.environ.get("TRT_CACHE_ROOT", PROJECT_ROOT / "trt_cache")).resolve()
SMPL_NEUTRAL_PATH = SMPL_FITTER_ROOT / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
SMPL_FACES_6890_PATH = SMPL_MODEL_ROOT / "smpl_faces_6890.npy"
VERTEX_SUBSET_1024_PATH = SMPL_FITTER_ROOT / "vertex_subset_1024.npz"
