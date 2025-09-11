@echo off
echo Setting up CUDA development environment...

REM Set Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" >nul 2>&1

REM Check if cl.exe is available
where cl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Visual Studio C++ compiler not found
    echo Please ensure Visual Studio 2022 with C++ tools is installed
    pause
    exit /b 1
)

REM Check if nvcc is available
where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: CUDA compiler not found
    echo Please ensure CUDA Toolkit is installed and in PATH
    pause
    exit /b 1
)

echo Compiling CUDA Mandelbrot program...
nvcc -o mandelbrot_cuda_clean mandelbrot_cuda_clean.cu

if %ERRORLEVEL% equ 0 (
    echo.
    echo ===== BUILD SUCCESSFUL =====
    echo Executable: mandelbrot_cuda_clean.exe
    echo.
    echo Ready to run!
) else (
    echo.
    echo ===== BUILD FAILED =====
    echo Error code: %ERRORLEVEL%
    echo Please check the error messages above
)

pause
