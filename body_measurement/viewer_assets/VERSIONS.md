# Viewer Assets Versions

`viewer_runtime.bundle.js` is the only browser runtime file loaded by `viewer.html`.

| Package | Version | License | Build use |
|---|---:|---|---|
| `esbuild` | `0.24.2` | MIT | Creates the classic offline bundle |
| `three` | `0.184.0` | MIT | Renderer, camera, material, controls |
| `three-mesh-bvh` | `0.9.10` | MIT | Raycast acceleration and mesh queries |

Rebuild command:

```powershell
python -m body_measurement.scripts.build_viewer_runtime
```

The delivered clothing workspace opens from `file://` and does not require npm, a dev server, or network access.
