"""BoxMOT compatibility helpers."""


def create_bytetrack(*args, **kwargs):
    candidates = (
        "boxmot",
        "boxmot.trackers.bytetrack.bytetrack",
        "boxmot.trackers.bbox.bytetrack.bytetrack",
    )
    last_error: ImportError | None = None
    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=["ByteTrack"])
            ByteTrack = getattr(module, "ByteTrack")
            break
        except (ImportError, AttributeError) as exc:
            last_error = ImportError(f"{module_name}: {exc}")
    else:
        raise ImportError("Could not import ByteTrack from supported boxmot layouts") from last_error

    return ByteTrack(*args, **kwargs)
