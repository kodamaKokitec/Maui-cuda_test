@echo off
setlocal enabledelayedexpansion

echo ====================================================
echo CUDA Build System - Environment Auto-Detection
echo ====================================================
echo.

REM ============================================================================
REM Visual Studio Detection
REM ============================================================================
echo [1/4] Detecting Visual Studio installation...

set "VS_FOUND=0"
set "VS_VERSION="
set "VS_PATH="

REM Check for Visual Studio 2022
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"
    set "VS_VERSION=2022 Professional"
    set "VS_FOUND=1"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
    set "VS_VERSION=2022 Community"
    set "VS_FOUND=1"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat" (
    set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat"
    set "VS_VERSION=2022 Enterprise"
    set "VS_FOUND=1"
)

REM Check for Visual Studio 2019 if 2022 not found
if !VS_FOUND! equ 0 (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\VsDevCmd.bat" (
        set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\VsDevCmd.bat"
        set "VS_VERSION=2019 Professional"
        set "VS_FOUND=1"
    ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" (
        set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
        set "VS_VERSION=2019 Community"
        set "VS_FOUND=1"
    ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\Common7\Tools\VsDevCmd.bat" (
        set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\Common7\Tools\VsDevCmd.bat"
        set "VS_VERSION=2019 Enterprise"
        set "VS_FOUND=1"
    )
)

if !VS_FOUND! equ 1 (
    echo [OK] Found Visual Studio !VS_VERSION!
    echo      Path: !VS_PATH!
) else (
    echo [ERROR] Visual Studio 2019/2022 not found
    echo Please install Visual Studio with C++ development tools
    pause
    exit /b 1
)

echo.

REM ============================================================================
REM CUDA Detection
REM ============================================================================
echo [2/4] Detecting CUDA installation...

set "CUDA_FOUND=0"
set "CUDA_VERSION="
set "CUDA_PATH_DETECTED="

REM Check for CUDA via nvcc command
where nvcc >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set "CUDA_FOUND=1"
    for /f "tokens=*" %%i in ('nvcc --version 2^>nul ^| findstr "release"') do (
        set "CUDA_VERSION=%%i"
    )
)

REM Check common CUDA installation paths
if !CUDA_FOUND! equ 0 (
    for %%v in (12.6 12.5 12.4 12.3 12.2 12.1 12.0 11.8 11.7) do (
        if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%v\bin\nvcc.exe" (
            set "CUDA_PATH_DETECTED=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%v"
            set "CUDA_FOUND=1"
            set "CUDA_VERSION=%%v"
            set "PATH=!CUDA_PATH_DETECTED!\bin;!PATH!"
            goto :cuda_found
        )
    )
)

:cuda_found
if !CUDA_FOUND! equ 1 (
    echo [OK] Found CUDA Toolkit
    if defined CUDA_VERSION echo      Version: !CUDA_VERSION!
    if defined CUDA_PATH_DETECTED echo      Path: !CUDA_PATH_DETECTED!
) else (
    echo [ERROR] CUDA Toolkit not found
    echo Please install CUDA Toolkit from NVIDIA website
    pause
    exit /b 1
)

echo.

REM ============================================================================
REM Setup Build Environment
REM ============================================================================
echo [3/4] Setting up build environment...

echo Setting up Visual Studio environment...
call "!VS_PATH!" -arch=amd64 >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Failed to setup Visual Studio environment
    pause
    exit /b 1
)

REM Verify compiler availability
where cl >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Visual Studio C++ compiler not available
    echo Please ensure Visual Studio C++ tools are installed
    pause
    exit /b 1
)

where nvcc >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [ERROR] CUDA compiler not available
    echo Please ensure CUDA Toolkit is properly installed
    pause
    exit /b 1
)

echo [OK] Build environment ready

echo.

REM ============================================================================
REM Build Process
REM ============================================================================
echo [4/4] Building CUDA programs...

set "BUILD_FAILED=0"

REM Build hello_cuda_new.cu
echo Building hello_cuda_new.cu...
nvcc -o hello_cuda_new.exe hello_cuda_new.cu
if !ERRORLEVEL! equ 0 (
    echo [OK] hello_cuda_new.exe built successfully
) else (
    echo [ERROR] Failed to build hello_cuda_new.cu
    set "BUILD_FAILED=1"
)

REM Build mandelbrot_cuda_clean_fixed.cu
echo Building mandelbrot_cuda_clean_fixed.cu...
nvcc -o mandelbrot_cuda_clean_fixed.exe mandelbrot_cuda_clean_fixed.cu
if !ERRORLEVEL! equ 0 (
    echo [OK] mandelbrot_cuda_clean_fixed.exe built successfully
) else (
    echo [ERROR] Failed to build mandelbrot_cuda_clean_fixed.cu
    set "BUILD_FAILED=1"
)

REM Build test_cuda_wrapper.cpp
echo Building test_cuda_wrapper.cpp...
cl /EHsc test_cuda_wrapper.cpp /Fe:test_cuda_wrapper_x64.exe
if !ERRORLEVEL! equ 0 (
    echo [OK] test_cuda_wrapper_x64.exe built successfully
) else (
    echo [ERROR] Failed to build test_cuda_wrapper.cpp
    set "BUILD_FAILED=1"
)

echo.

REM ============================================================================
REM Build Summary
REM ============================================================================
echo ====================================================
echo Build Summary
echo ====================================================

if !BUILD_FAILED! equ 0 (
    echo [SUCCESS] All builds completed successfully!
    echo.
    echo Available executables:
    echo   - hello_cuda_new.exe               (Simple CUDA test)
    echo   - mandelbrot_cuda_clean_fixed.exe  (Mandelbrot visualization)
    echo   - test_cuda_wrapper_x64.exe        (CUDA wrapper test)
    echo.
    echo You can now run any of these programs.
) else (
    echo [WARNING] Some builds failed. Check error messages above.
)

echo.
echo Build environment details:
echo   Visual Studio: !VS_VERSION!
echo   CUDA: !CUDA_VERSION!
echo   Architecture: x64
echo.

pause
