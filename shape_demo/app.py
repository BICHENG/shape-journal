"""Serve the Shape Journal viewer and a local, append-only capture archive."""

import argparse
from datetime import date
from functools import lru_cache
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from body_measurement.shape_tracking import ShapeObservation
from core.motionpack import NoPersonTrackError, load_artifacts, reconstruct_subject
from core.motionpack.io import _file_sha256
from video_io import VideoDecodeError
from shape_demo.alignment import source_alignment, source_motion
from shape_demo.build import (
    DEFAULT_PACK, DEFAULT_VIDEO, ROOT, archive_mesh, bundle_scene, demo_mesh, mesh_payload,
    positive_height, render_preview, visit_key,
)
from shape_demo.mock import MOCK_INDEX, group_payload, mock_payload

VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
DEFAULT_ARCHIVE = ROOT / "projects/shape-tracer/archive.json"
LOG = logging.getLogger(__name__)
LIVE = {"Cache-Control": "no-store"}


def load_timeline(archive):
    canonical, mesh = archive_mesh(archive)
    data = mesh_payload(*mesh)
    data["mode"] = "archive"
    return canonical, data


def create_app(archive_path: Path = DEFAULT_ARCHIVE, modules: Path | None = None,
               mock_index: Path = MOCK_INDEX) -> FastAPI:
    archive_path = archive_path.resolve()
    capture_root = archive_path.with_suffix(".captures")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    mock_index = mock_index.resolve()
    subjects = json.loads(mock_index.read_bytes())["subjects"] if mock_index.is_file() else {}
    subjects = dict(sorted(subjects.items(), key=lambda entry: entry[1]["name"].casefold()))
    default_user = "bi-cheng" if "bi-cheng" in subjects else next(iter(subjects), "")

    def demo_user(user):
        key = user or default_user
        if key and key not in subjects:
            raise HTTPException(404, "This sample was not found.")
        return key

    @lru_cache(maxsize=len(subjects) or 1)
    def demo_profile(user):
        if user:
            data, motion = mock_payload(mock_index, user, subjects[user])
        else:
            if not DEFAULT_PACK.exists():
                reconstruct_subject(DEFAULT_VIDEO, DEFAULT_PACK)
            pack = load_artifacts(DEFAULT_PACK)
            body = pack.body
            if body.source.video_sha256 != _file_sha256(DEFAULT_VIDEO):
                raise ValueError("The default BodyPack does not match bc1.mp4")
            data = mesh_payload(*demo_mesh(body, 170.0))
            data["mode"] = "demo"
            data["source"], pipeline = source_alignment(DEFAULT_VIDEO, pack, tail=True)
            motion = source_motion(DEFAULT_VIDEO, body, data["source"],
                                   DEFAULT_PACK / f"aligned-v6-{_file_sha256(body.path)[:16]}.bin",
                                   pipeline) if data["source"] else None
        if motion and motion.is_file():
            data["source"]["motion"] = "/source-motion" + (f"?user={user}" if user else "")
        return data, motion

    template = bundle_scene(modules)

    empty = {"subject_id": "local-person", "height_cm": 170.0, "visits": []}
    snapshot = load_timeline(json.loads(archive_path.read_text(encoding="utf-8-sig"))) if archive_path.exists() else (empty, None)
    writer = Lock()
    progress = {"step": "idle"}
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    group_ready = len(subjects) > 1 and all(
        (mock_index.parent / key / "excerpt" / "source.json").is_file()
        for key in subjects
    )

    @lru_cache(maxsize=1)
    def demo_group():
        return group_payload(mock_index, ((key, demo_profile(key)[0]) for key in subjects))

    def page(view, selected=None, user=None):
        observed, timeline = snapshot
        visits = [
            {field: record[field] for field in ("key", "date", "device", "count")}
            for record in timeline["records"]
        ] if timeline else []
        intake = {"view": view, "visits": visits, "height_cm": observed["height_cm"], "record": selected}
        if view == "archive":
            records = []
            for record, visit in zip(timeline["records"], observed["visits"], strict=True):
                takes = [item["video_sha256"] for item in visit["captures"]]
                records.append({**record, "takes": [
                    key if (capture_root / key / "source.json").is_file() else None for key in takes
                ]})
            data = {**timeline, "records": records}
        else:
            data = demo_group() if view == "group" else demo_profile(user)[0]
            intake["group"] = group_ready
        return HTMLResponse(render_preview(template, data, intake), headers=LIVE)

    @app.get("/")
    def home(user: str | None = None, add: bool = False):
        if group_ready and not (user or add):
            return page("group")
        return page("demo", user=demo_user(user))

    @app.get("/archive")
    def history(visit: str | None = None):
        if snapshot[1] is None:
            return RedirectResponse("/?add=1")
        return page("archive", visit)

    @app.get("/api/status")
    def status():
        return JSONResponse(progress, headers=LIVE)

    @app.get("/source-motion")
    def aligned_motion(user: str | None = None, excerpt: bool = False):
        key = demo_user(user)
        if excerpt:
            motion = mock_index.parent / key / "excerpt" / "motion-v6.bin"
        else:
            _, motion = demo_profile(key)
        if motion is None or not motion.is_file():
            raise HTTPException(404, "Playback is unavailable for this clip.")
        return FileResponse(motion, media_type="application/octet-stream")

    def archived_take(key):
        if not any(key == item["video_sha256"] for visit in snapshot[0]["visits"] for item in visit["captures"]):
            raise HTTPException(404, "This clip was not found.")
        directory = capture_root / key
        if not (directory / "source.json").is_file():
            raise HTTPException(404, "The original video is unavailable.")
        return directory

    @app.get("/api/captures/{key}")
    def take_source(key: str):
        directory = archived_take(key)
        source = json.loads((directory / "source.json").read_bytes())
        if source["count"] > 1 and (directory / "motion-v6.bin").is_file():
            source["motion"] = f"/api/captures/{key}/motion"
        return JSONResponse(source, headers=LIVE)

    @app.get("/api/captures/{key}/motion")
    def take_motion(key: str):
        path = archived_take(key) / "motion-v6.bin"
        if not path.is_file():
            raise HTTPException(404, "Playback is unavailable for this clip.")
        return FileResponse(path, media_type="application/octet-stream", headers=LIVE)

    def append(form):
        nonlocal snapshot, progress
        observed = snapshot[0]
        visits = observed["visits"]
        key = form.get("visit", "")
        selected = next((visit for visit in visits if visit_key(visit) == key), None)
        if key and selected is None:
            raise HTTPException(400, "This visit was not found. Refresh and try again.")
        if selected is None:
            try:
                day = date.fromisoformat(str(form.get("date", ""))).isoformat()
                height = observed["height_cm"] if visits else positive_height(float(form.get("height") or 170))
            except ValueError:
                raise HTTPException(400, "Check the date and height.") from None
            device = str(form.get("device", "")).strip() or "默认拍摄"
            if len(device) > 48:
                raise HTTPException(400, "Use up to 48 characters for the camera name.")
            selected = {"id": uuid4().hex, "date": day, "protocol_id": device, "captures": []}
        else:
            height = observed["height_cm"]

        files = form.getlist("videos")
        if not files or any(not isinstance(file, UploadFile) or Path(file.filename or "").suffix.lower() not in VIDEO_TYPES
                            or not file.size for file in files):
            raise HTTPException(400, "Choose a full-body video that can be played.")
        if any(file.size > 512 * 1024 * 1024 for file in files):
            raise HTTPException(400, "Each clip must be under 512 MB.")
        known = {item["video_sha256"]: visit for visit in visits for item in visit["captures"]}
        duplicates = 0
        with TemporaryDirectory(prefix=".intake-", dir=archive_path.parent) as directory:
            work = Path(directory)
            pending, existing, seen = [], [], set()
            for index, file in enumerate(files, 1):
                source = work / (str(index) + Path(file.filename).suffix.lower())
                with source.open("wb") as target:
                    while block := file.file.read(1024 * 1024):
                        target.write(block)
                fingerprint = _file_sha256(source)
                if fingerprint in seen:
                    duplicates += 1
                    continue
                seen.add(fingerprint)
                if fingerprint in known:
                    existing.append(known[fingerprint])
                    duplicates += 1
                else:
                    pending.append((source, fingerprint))

            if existing:
                if not key and not pending and all(item is existing[0] for item in existing):
                    if (existing[0]["date"], existing[0]["protocol_id"]) == (selected["date"], selected["protocol_id"]):
                        return {"visit": visit_key(existing[0]), "added": 0, "duplicates": duplicates}
                if any(item is not selected for item in existing):
                    raise HTTPException(409, "A clip already belongs to another visit. Select that visit or remove the clip.")
            if not pending:
                return {"visit": visit_key(selected), "added": 0, "duplicates": duplicates}

            captures = list(selected["captures"])
            staged_sources = []
            for index, (source, expected) in enumerate(pending, 1):
                def report(step):
                    nonlocal progress
                    progress = {"step": step, "clip": index, "total": len(pending)}

                fitted = reconstruct_subject(source, work / f"pack-{index}", on_stage=report)
                observation = ShapeObservation.from_body(
                    fitted, subject_id=observed["subject_id"], visit_id=selected["id"], protocol_id=selected["protocol_id"],
                )
                if observation.video_sha256 != expected:
                    raise ValueError("Source changed during inference")
                captures.append(json.loads(observation.to_json()))
                report("alignment")
                aligned, layers = source_alignment(source, load_artifacts(fitted.path))
                if not aligned or not layers:
                    raise ValueError("The new observation has no aligned inference frames")
                staged = work / expected
                staged.mkdir()
                if aligned["count"] > 1:
                    replay = source_motion(source, fitted, aligned, staged / "motion-v6.bin", layers)
                    if replay is None:
                        raise HTTPException(422, "Could not prepare playback. Check FFmpeg and try an MP4.")
                (staged / "source.json").write_text(json.dumps(aligned, separators=(",", ":")), encoding="utf-8")
                staged_sources.append(staged)

            progress = {"step": "saving"}
            updated = {**selected, "captures": captures}
            revised = [updated if visit is selected else visit for visit in visits] if key else [*visits, updated]
            candidate = load_timeline({**observed, "height_cm": height, "visits": revised})
            staging = work / "archive.json"
            staging.write_text(json.dumps(candidate[0], ensure_ascii=False, indent=2), encoding="utf-8")
            for staged in staged_sources:
                target = capture_root / staged.name
                target.mkdir(parents=True, exist_ok=True)
                for file in staged.iterdir():
                    file.replace(target / file.name)
            staging.replace(archive_path)
            snapshot = candidate
            return {"visit": visit_key(updated), "added": len(pending), "duplicates": duplicates}

    @app.post("/api/captures")
    async def upload(request: Request):
        nonlocal progress
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Upload from Shape Journal.")
        if not writer.acquire(blocking=False):
            raise HTTPException(409, "A video is processing. Try again when it finishes.")
        progress = {"step": "upload"}
        try:
            async with request.form() as form:
                return await run_in_threadpool(append, form)
        except (VideoDecodeError, NoPersonTrackError):
            raise HTTPException(422, "Use a clip with one person, with their head and feet in frame.") from None
        except HTTPException:
            raise
        except Exception:
            LOG.exception("Capture batch failed")
            raise HTTPException(500, "This visit could not be saved. Try the original clips again.") from None
        finally:
            progress = {"step": "idle"}
            writer.release()

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7886)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--mock-index", type=Path, default=MOCK_INDEX)
    parser.add_argument("--node-modules", type=Path)
    parser.add_argument("--open-browser", action="store_true", help="Open the local Library in your browser")
    args = parser.parse_args()
    host = "127.0.0.1"
    if not args.open_browser:
        uvicorn.run(create_app(args.archive, args.node_modules, args.mock_index), host=host, port=args.port)
        return

    from http.client import HTTPConnection, HTTPException as HTTPClientError
    from threading import Thread
    import webbrowser

    url = f"http://{host}:{args.port}/"
    connection = HTTPConnection(host, args.port, timeout=20)
    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        if response.status != 200 or b"<title>Shape Journal</title>" not in response.read(1024):
            parser.error(f"Port {args.port} is already in use by a different app.")
    except ConnectionRefusedError:
        pass
    except (OSError, HTTPClientError):
        parser.error(f"Port {args.port} did not respond with the Shape Journal Library.")
    else:
        webbrowser.open(url)
        return
    finally:
        connection.close()

    class BrowserServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started:
                Thread(target=webbrowser.open, args=(url,), daemon=True).start()

    config = uvicorn.Config(create_app(args.archive, args.node_modules, args.mock_index), host=host, port=args.port)
    server = BrowserServer(config)
    server.run()
    if not server.started:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
