@echo off
REM ==============================================================================
REM run.bat - Windows Single-Video Inference Entry Point
REM Usage: run.bat <video.mp4> <out.json> [out_highlight.mp4]
REM ==============================================================================

if "%~2"=="" (
    echo Usage: %0 ^<video.mp4^> ^<out.json^> [out_highlight.mp4]
    exit /b 1
)

set VIDEO_PATH=%~1
set OUTPUT_JSON=%~2
set HIGHLIGHT_MP4=%~3

echo [ActionAware] Processing: %VIDEO_PATH%
echo [ActionAware] Target JSON: %OUTPUT_JSON%

if not "%HIGHLIGHT_MP4%"=="" (
    python code\main.py --video "%VIDEO_PATH%" --output "%OUTPUT_JSON%" --highlight-mp4 "%HIGHLIGHT_MP4%" --config config\config.yaml
) else (
    python code\main.py --video "%VIDEO_PATH%" --output "%OUTPUT_JSON%" --config config\config.yaml
)

echo [ActionAware] Inference completed.
