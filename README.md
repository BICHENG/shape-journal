# Shape Journal

Shape Journal reconstructs a body shape from a video. It saves shapes by visit so users can compare measurements in a common standing pose and return to the original frame to check the fit.

## Run locally on Windows

Install Python 3.12 x64, Node.js 22 or newer and FFmpeg. Video reconstruction uses an NVIDIA GPU with a driver that supports CUDA 12.6. Set up the Python environment in the repository root.

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows_cuda.ps1
~~~

Place the model files you have access to in the following locations.

| Directory | Files |
| --- | --- |
| models/detector/yolo26/ | yolo26m.pt |
| weights/main/ | backbone.pth, backbone_buffers.pth, buffers.pth, field.pth, head.pth |
| output/ | canonical_smpl.npy |
| models/smpl_models/ | smpl_faces_6890.npy |
| models/smpl_models/smpl/ | basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl, kid_template.npy, vertex_subset_1024.npz, vertex_subset_joint_regr_post_lbs_1024.npy |

Place a full-body clip of one person at test_data/bc1.mp4. Double-click [open_shape_journal.bat](open_shape_journal.bat) to open the local app. The first reconstruction can take longer while TensorRT compiles the models.

Open **Add Video** to start a visit or add a repeat capture. The app saves visits in projects/shape-tracer/archive.json. Select a date to compare shapes and measurements. Press Ctrl+C in the launch window to stop the server.

## Check the source

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
node --test body_measurement/viewer_assets/src/surface_measurement.test.js
~~~

To build the standalone measurement viewer from its source, run the viewer build once.

~~~powershell
.\.venv\Scripts\python.exe -m body_measurement.scripts.build_viewer_runtime
~~~
