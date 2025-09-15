@echo off
setlocal enabledelayedexpansion

echo ====================================================
echo CUDA Debug Build System
echo ====================================================
echo.

REM Parse command line arguments
set "BUILD_TYPE=DEBUG"
set "VERBOSE=0"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--release" set "BUILD_TYPE=RELEASE"
if /i "%~1"=="--debug" set "BUILD_TYPE=DEBUG"
if /i "%~1"=="--verbose" set "VERBOSE=1"
if /i "%~1"=="-v" set "VERBOSE=1"
shift
goto :parse_args
:args_done

echo Build Type: !BUILD_TYPE!
if !VERBOSE! equ 1 echo Verbose Mode: ON
echo.

REM Include the environment detection from build_all.bat
call :detect_environment
if !ERRORLEVEL! neq 0 exit /b 1

echo [4/4] Building CUDA programs in !BUILD_TYPE! mode...

REM Set build flags based on build type
if "!BUILD_TYPE!"=="DEBUG" (
    set "NVCC_FLAGS=-g -G -O0 -DDEBUG"
    set "CL_FLAGS=/Od /Zi /DDEBUG"
    echo Debug flags: NVCC=!NVCC_FLAGS!, CL=!CL_FLAGS!
) else (
    set "NVCC_FLAGS=-O3 -DNDEBUG"
    set "CL_FLAGS=/O2 /DNDEBUG"
    echo Release flags: NVCC=!NVCC_FLAGS!, CL=!CL_FLAGS!
)

set "BUILD_FAILED=0"

REM Build hello_cuda_new.cu
echo Building hello_cuda_new.cu [!BUILD_TYPE!]...
if !VERBOSE! equ 1 (
    nvcc !NVCC_FLAGS! -o hello_cuda_new_!BUILD_TYPE!.exe hello_cuda_new.cu
) else (
    nvcc !NVCC_FLAGS! -o hello_cuda_new_!BUILD_TYPE!.exe hello_cuda_new.cu >nul 2>&1
)
if !ERRORLEVEL! equ 0 (
    echo [OK] hello_cuda_new_!BUILD_TYPE!.exe built successfully
) else (
    echo [ERROR] Failed to build hello_cuda_new.cu
    set "BUILD_FAILED=1"
)

REM Build mandelbrot_cuda_clean_fixed.cu
echo Building mandelbrot_cuda_clean_fixed.cu [!BUILD_TYPE!]...
if !VERBOSE! equ 1 (
    nvcc !NVCC_FLAGS! -o mandelbrot_cuda_clean_fixed_!BUILD_TYPE!.exe mandelbrot_cuda_clean_fixed.cu
) else (
    nvcc !NVCC_FLAGS! -o mandelbrot_cuda_clean_fixed_!BUILD_TYPE!.exe mandelbrot_cuda_clean_fixed.cu >nul 2>&1
)
if !ERRORLEVEL! equ 0 (
    echo [OK] mandelbrot_cuda_clean_fixed_!BUILD_TYPE!.exe built successfully
) else (
    echo [ERROR] Failed to build mandelbrot_cuda_clean_fixed.cu
    set "BUILD_FAILED=1"
)

REM Build test_cuda_wrapper.cpp
echo Building test_cuda_wrapper.cpp [!BUILD_TYPE!]...
if !VERBOSE! equ 1 (
    cl !CL_FLAGS! /EHsc test_cuda_wrapper.cpp /Fe:test_cuda_wrapper_!BUILD_TYPE!.exe
) else (
    cl !CL_FLAGS! /EHsc test_cuda_wrapper.cpp /Fe:test_cuda_wrapper_!BUILD_TYPE!.exe >nul 2>&1
)
if !ERRORLEVEL! equ 0 (
    echo [OK] test_cuda_wrapper_!BUILD_TYPE!.exe built successfully
) else (
    echo [ERROR] Failed to build test_cuda_wrapper.cpp
    set "BUILD_FAILED=1"
)

echo.
echo ====================================================
echo Debug Build Summary
echo ====================================================

if !BUILD_FAILED! equ 0 (
    echo [SUCCESS] All !BUILD_TYPE! builds completed successfully!
    echo.
    echo Available executables:
    echo   - hello_cuda_new_!BUILD_TYPE!.exe
    echo   - mandelbrot_cuda_clean_fixed_!BUILD_TYPE!.exe
    echo   - test_cuda_wrapper_!BUILD_TYPE!.exe
    echo.
    if "!BUILD_TYPE!"=="DEBUG" (
        echo Debug symbols included - ready for debugging with:
        echo   - Visual Studio Debugger
        echo   - NVIDIA Nsight Compute
        echo   - CUDA-GDB
    ) else (
        echo Optimized for performance
    )
) else (
    echo [WARNING] Some builds failed. Use --verbose flag for details.
)

echo.
pause
exit /b 0

REM ============================================================================
REM Environment Detection Function
REM ============================================================================
:detect_environment
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
) else (
    echo [ERROR] Visual Studio 2019/2022 not found
    exit /b 1
)

echo [2/4] Detecting CUDA installation...
where nvcc >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo [OK] Found CUDA Toolkit
) else (
    echo [ERROR] CUDA Toolkit not found
    exit /b 1
)

echo [3/4] Setting up build environment...
call "!VS_PATH!" -arch=amd64 >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Failed to setup Visual Studio environment
    exit /b 1
)
echo [OK] Build environment ready
echo.

exit /b 0
