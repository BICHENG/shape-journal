"""CLI wrapper for building or inspecting MotionPack base data.

The library implementation lives in ``core.motionpack``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _print_detection_info(det) -> None:
    print(f"视频: {det.video_path}")
    print(f"帧数: {det.total_frames}, FPS: {det.source_fps}, 尺寸: {det.shape}")
    print(f"Tracks ({len(det.tracks)}), 按帧数排序:")
    for tid, track in sorted(det.tracks.items(), key=lambda item: -len(item[1].frames)):
        coverage = len(track.frames) / det.total_frames * 100
        avg_area = (track.bboxes[:, 2] * track.bboxes[:, 3]).mean()
        print(f"  T{tid}: {len(track.frames)}帧 ({coverage:.1f}%), 平均面积={avg_area:.0f}, 帧{track.frames[0]}-{track.frames[-1]}")


def _print_pose_info(pose) -> None:
    print(f"Pose tracks ({len(pose.tracks)}):")
    for tid, track in pose.tracks.items():
        print(f"  {tid}: {len(track.poses3d)} 帧, poses3d shape={track.poses3d.shape}")


def _print_body_info(body) -> None:
    print(f"Body tracks ({len(body.tracks)}):")
    for tid, track in body.tracks.items():
        print(f"  {tid}: betas={track.betas[:3]}..., {len(track.joints)} 帧")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build or inspect MotionPack base data")
    parser.add_argument("command", choices=["det", "pose", "body", "all", "info"])
    parser.add_argument("--video", "-v", help="视频文件路径")
    parser.add_argument("--det", "-d", help="Detection文件路径")
    parser.add_argument("--pose", "-p", help="Pose文件路径")
    parser.add_argument("--body", "-b", help="Body文件路径")
    parser.add_argument("--tracks", "-t", type=int, nargs="+", help="指定track IDs")
    parser.add_argument("--out", "-o", help="输出路径，all 接收新目录")
    args = parser.parse_args()

    if args.command == "info":
        from core.motionpack import load_body, load_detection, load_pose

        for path, load, describe in (
            (args.det, load_detection, _print_detection_info),
            (args.pose, load_pose, _print_pose_info),
            (args.body, load_body, _print_body_info),
        ):
            if path:
                describe(load(path))
                return 0
        parser.error("info requires --det, --pose, or --body")

    from core.motionpack import Body, Detection, Pose, build_artifacts

    source_arg = {"det": "video", "pose": "det", "body": "pose", "all": "video"}[args.command]
    source = getattr(args, source_arg)
    if not source:
        parser.error(f"{args.command} requires --{source_arg}")
    if args.command == "all":
        output = args.out or Path(source).with_suffix(".motionpack")
        result = build_artifacts(source, output, track_ids=args.tracks, max_tracks=1)
        print(f"完成: {result['body_path']}")
        return 0

    prior_suffix = ".mp4" if source_arg == "video" else f".{source_arg}"
    suffix = f".{args.command}"
    output = args.out or (source.removesuffix(prior_suffix) + suffix)
    if args.command == "det":
        Detection(source).save(output)
    elif args.command == "pose":
        Pose(Detection.load(source), args.tracks).save(output, det_path=source)
    else:
        Body(Pose.load(source), args.tracks).save(output, pose_path=source)
    print(f"已保存: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
