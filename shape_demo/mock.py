"""Prepare the local Shape Journal demo users from a directory of source videos."""

import argparse
from base64 import b64encode
from datetime import date, timedelta
import json
import os
from pathlib import Path
import random
import re

from core.motionpack import load_artifacts, load_body, reconstruct_subject
from core.motionpack.io import _file_sha256
from core.output_directory import new_output_directory
from shape_demo.alignment import MEASURE_TAIL_SECONDS, source_alignment, source_motion
from shape_demo.build import DEFAULT_PACK, DEFAULT_VIDEO, ROOT, demo_mesh, mesh_payload


MOCK_INDEX = ROOT / "projects/shape-tracer/users-mock/index.json"


def mock_payload(index: Path, key: str, subject: dict, *, stl_dir=None):
    capture = index.parent / key / "capture"
    body = load_body(index.parent / subject["body"])
    if body.source.video_sha256 != subject["video_sha256"]:
        raise ValueError(f"BodyPack does not match {subject['name']}")
    data = mesh_payload(*demo_mesh(body, 170, subject["dates"], key), stl_dir=stl_dir)
    data.update(mode="demo", subject=subject["name"],
                source=json.loads((capture / "source.json").read_bytes()))
    return data, capture / "motion-v6.bin"


def group_payload(index: Path, profiles, *, embedded=False):
    people = []
    fields = ("pose_root", "positions", "rig", "records", "views", "tape") if embedded else ("pose_root", "positions", "rig")
    for key, data in profiles:
        excerpt = index.parent / key / "excerpt"
        source = json.loads((excerpt / "source.json").read_bytes())
        if embedded:
            packed = b64encode((excerpt / "motion-v6.bin").read_bytes()).decode("ascii")
            source["motion"] = "data:application/octet-stream;base64," + packed
        else:
            source["motion"] = f"/source-motion?user={key}&excerpt=true"
        people.append({"id": key, "name": data["subject"], "source": source, **{field: data[field] for field in fields}})
    return {"triangles": data["triangles"], "people": people}


def prepare(videos: Path, index: Path, through: date):
    start = date(2026, 10, 1)
    index = index.resolve()
    index.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(index.read_bytes()) if index.is_file() else {"subjects": {}}
    subjects = manifest["subjects"]
    bc1_sha = _file_sha256(DEFAULT_VIDEO)
    clips = sorted(videos.resolve(strict=True).glob("*.mp4"), key=lambda video: video.name.casefold())
    if not clips:
        raise ValueError("No MP4 files found in the demo input directory")

    entries = [(re.sub(r"[^a-z0-9]+", "-", clip.stem.lower()).strip("-"), clip) for clip in clips]
    keys = [key for key, _ in entries]
    if not all(keys) or len(keys) != len(set(keys)):
        raise ValueError("Use distinct demo filenames with letters or numbers")
    for key, video in entries:
        fingerprint = _file_sha256(video)
        capture_dir = index.parent / key
        pack_dir = DEFAULT_PACK if fingerprint == bc1_sha and DEFAULT_PACK.is_dir() else capture_dir / "motionpack"
        if key in subjects:
            if subjects[key]["video_sha256"] != fingerprint:
                raise ValueError(f"The recorded source for {video.stem} has changed")
            continue
        generator = random.Random(f"ShapeTracer:{key}:{through.isoformat()}")
        days = sorted(generator.sample(range((through - start).days + 1), k=8))
        subjects[key] = {
            "name": video.stem, "video_sha256": fingerprint,
            "dates": [(start + timedelta(days=offset)).isoformat() for offset in days],
            "body": os.path.relpath(pack_dir / "body.body", index.parent),
        }

    for key, video in entries:
        subject = subjects[key]
        capture_dir = index.parent / key
        capture_dir.mkdir(exist_ok=True)
        pack_path = (index.parent / subject["body"]).resolve()
        print(f"\n{subject['name']}", flush=True)
        if not pack_path.is_file():
            reconstruct_subject(video, pack_path.parent, on_stage=lambda stage: print(f"  {stage}", flush=True))
        pack = load_artifacts(pack_path)
        if pack.body.source.video_sha256 != subject["video_sha256"]:
            raise ValueError(f"BodyPack does not match {video.name}")
        media = capture_dir / "capture"
        source_file = media / "source.json"
        if source_file.is_file():
            source = json.loads(source_file.read_bytes())
            tail_frame = source["first"] + max(0, source["count"] - round(MEASURE_TAIL_SECONDS * source["fps"]))
            if source.get("measure_start") != tail_frame or source["frame"] != tail_frame:
                source, _ = source_alignment(video, pack, tail=True)
                metadata = source_file.with_suffix(".tmp")
                metadata.write_text(json.dumps(source, separators=(",", ":")), encoding="utf-8")
                metadata.replace(source_file)
        else:
            with new_output_directory(media) as output:
                source, pipeline = source_alignment(video, pack, tail=True)
                if source is None or pipeline is None:
                    raise ValueError(f"No tracked reconstruction for {video.name}")
                if source["count"] > 1 and source_motion(video, pack.body, source, output / "motion-v6.bin", pipeline) is None:
                    raise ValueError(f"Could not prepare aligned video for {video.name}")
                (output / "source.json").write_text(json.dumps(source, separators=(",", ":")), encoding="utf-8")
        source = json.loads(source_file.read_bytes())
        excerpt = capture_dir / "excerpt"
        if not (excerpt / "source.json").is_file():
            with new_output_directory(excerpt) as output:
                window, frames = source_alignment(video, pack, excerpt=True)
                if source_motion(video, pack.body, window, output / "motion-v6.bin", frames) is None:
                    raise ValueError(f"Could not prepare the group replay for {video.name}")
                (output / "source.json").write_text(json.dumps(window, separators=(",", ":")), encoding="utf-8")
        print(f"  Ready: {source['count']}/{source['total']} tracked frames, "
              f"{len(subject['dates'])} dates, {source['width']}×{source['height']}", flush=True)

    pending = index.with_suffix(".tmp")
    pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path)
    parser.add_argument("--through", required=True, type=date.fromisoformat)
    parser.add_argument("--index", type=Path, default=MOCK_INDEX)
    args = parser.parse_args()
    prepare(args.videos, args.index, args.through)
