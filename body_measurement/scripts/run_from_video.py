"""Build a portable workbench from a video or a saved BodyPack."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from uuid import uuid4

from body_measurement.result_package import (
    RECORDS_ARTIFACT,
    MeasurementSnapshot,
    _snapshot_files,
    _validate_height,
    build_workbench,
)
from core.motionpack import MotionPackBody, load_body
from core.output_directory import new_output_directory


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "projects" / "measurement-batch"


def run_measurement(output_dir: Path, height_cm: float | None, **inputs) -> dict:
    package, _ = run_batch(output_dir, height_cm, **inputs)
    package.pop("manifest")
    return package


def run_batch(
    output_dir: Path,
    height_cm: float | None,
    *,
    video: Path | None = None,
    body: Path | MotionPackBody | None = None,
    track_id: int | None = None,
    source_identity: dict | None = None,
    reference_measurements: list[dict] | None = None,
    public_item_ids: list[str] | None = None,
) -> tuple[dict, MeasurementSnapshot]:
    if (video is None) == (body is None):
        raise ValueError("Provide exactly one video or BodyPack")
    _validate_height(height_cm)
    output = Path(output_dir).absolute()
    with new_output_directory(output) as run:
        if video is not None:
            from core.motionpack import reconstruct_subject

            body = reconstruct_subject(video, run / "motionpack", track_id=track_id)
        elif not isinstance(body, MotionPackBody):
            body = load_body(body)
        workspace = run / "workbench"
        package = build_workbench(
            body, workspace, height_cm=height_cm, track_id=track_id,
            source_identity=source_identity, reference_measurements=reference_measurements,
            public_item_ids=public_item_ids, include_snapshot=True,
        )
        # Consume the files produced by this run while the enclosing directory is private.
        snapshot = _snapshot_files(package)

    workspace = output / "workbench"
    paths = package["artifacts"]
    package.update(package_dir=str(workspace), manifest_path=str(workspace / paths["manifest"]))
    return package, replace(
        snapshot, workspace=workspace.resolve(), progress_path=(workspace / paths[RECORDS_ARTIFACT]).resolve(),
        csv_path=(workspace / paths["csv"]).resolve(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="从视频或已有 BodyPack 生成量体工作台和当前量体表")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="单人全身视频")
    source.add_argument("--body", type=Path, help="已有 BodyPack")
    parser.add_argument("--height", required=True, type=float, help="身高（cm）")
    parser.add_argument("--track-id", type=int, help="指定人物轨迹")
    parser.add_argument("--output-dir", type=Path, help="本次计算的新目录")
    args = parser.parse_args()
    output = args.output_dir or DEFAULT_OUTPUT_ROOT / f"{(args.video or args.body).stem}-{uuid4().hex[:8]}"
    try:
        result = run_measurement(output, args.height, video=args.video, body=args.body, track_id=args.track_id)
    except Exception as exc:
        print(f"量体失败。{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
