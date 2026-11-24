from pathlib import Path
from collections.abc import Callable
from tempfile import TemporaryDirectory
import warnings

import torch


def cached_trt(
    path: Path, make_model: Callable[[], torch.nn.Module], example: torch.Tensor, *, precision: torch.dtype,
) -> torch.nn.Module:
    import torch_tensorrt

    if path.is_file():
        try:
            with path.open("rb") as stream:
                graph = torch.export.load(stream).module()
            with torch.inference_mode():
                graph(example)
            torch.cuda.synchronize(example.device)
            return graph
        except Exception as error:
            warnings.warn(f"TensorRT cache {path.name} cannot be loaded: {error}")

    graph = torch_tensorrt.compile(
        make_model(), ir="dynamo", inputs=[example], enabled_precisions={precision},
        workspace_size=1 << 30, truncate_long_and_double=True, use_explicit_typing=False,
    )
    with torch.inference_mode():
        graph(example)
    torch.cuda.synchronize(example.device)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(dir=path.parent) as directory:
            pending = Path(directory) / path.name
            # Python opens Unicode paths that the native serializer cannot open on Windows.
            with pending.open("wb") as stream:
                torch_tensorrt.save(graph, stream, output_format="exported_program")
            pending.replace(path)
    except Exception as error:
        warnings.warn(f"TensorRT cache {path.name} cannot be saved: {error}")
    return graph
