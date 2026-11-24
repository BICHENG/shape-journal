$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    python -m venv (Join-Path $project '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Creating .venv failed.' }
}

$pytorch = 'https://download.pytorch.org/whl/cu126'
& $python -m pip install --extra-index-url $pytorch --extra-index-url 'https://pypi.nvidia.com' `
    --constraint (Join-Path $PSScriptRoot 'windows-cu126.constraints.txt') `
    -r (Join-Path $project 'requirements.txt') 'tensorrt'
if ($LASTEXITCODE -ne 0) { throw 'Installing CUDA runtime dependencies failed.' }

& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed Python dependencies are inconsistent.' }

& $python -c "import torch, torchvision, tensorrt, torch_tensorrt; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(), torch.__version__, tensorrt.__version__, torch_tensorrt.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'CUDA environment verification failed.' }
