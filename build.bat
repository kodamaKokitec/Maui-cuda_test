@echo off
echo CUDA Mandelbrot Set Visualization - Build Script
echo ===============================================

REM CUDA Toolkitのパスを確誁E
where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: CUDA Toolkit not found in PATH
    echo Please install CUDA Toolkit and add it to your PATH
    echo Typical installation path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
    pause
    exit /b 1
)

REM Visual Studio環墁E�E確誁E
where cl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Visual Studio C++ compiler not found
    echo Please run this script from Visual Studio Developer Command Prompt
    echo Or install Visual Studio with C++ development tools
    pause
    exit /b 1
)

echo Building CUDA program...
echo.

REM CUDAプログラムをコンパイル
nvcc -o mandelbrot_cuda mandelbrot_cuda.cu

if %ERRORLEVEL% equ 0 (
    echo.
    echo Build successful!
    echo Executable: mandelbrot_cuda.exe
    echo.
    echo Run 'mandelbrot_cuda.exe' to execute the program
) else (
    echo.
    echo Build failed!
    echo Please check the error messages above
    pause
    exit /b 1
)

pause
