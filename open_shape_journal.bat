@echo off
setlocal
cd /d "%~dp0" || exit /b 1
title Shape Journal
if not exist ".venv\Scripts\python.exe" (
    echo Run scripts\setup_windows_cuda.ps1 first.
    pause
    exit /b 1
)
if not defined TRT_CACHE_ROOT set "TRT_CACHE_ROOT=%CD%\trt_cache\windows-cu126"
if not defined npm_config_prefer_offline set "npm_config_prefer_offline=true"
".venv\Scripts\python.exe" -m shape_demo.app --port 7911 --open-browser %*
if errorlevel 1 (
    pause
    exit /b 1
)
