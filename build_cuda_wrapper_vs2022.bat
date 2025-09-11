@echo off
echo Building CUDA Mandelbrot Wrapper for MAUI with VS2022 Build Tools...

REM Check if CUDA is available
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
if not exist "%CUDA_PATH%" (
    echo Error: CUDA not found at %CUDA_PATH%
    pause
    exit /b 1
)

REM Set up Visual Studio 2022 Build Tools environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if %ERRORLEVEL% neq 0 (
    echo Trying VS2022 Enterprise...
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
)
if %ERRORLEVEL% neq 0 (
    echo Trying VS2022 Professional...
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
)
if %ERRORLEVEL% neq 0 (
    echo Trying VS2022 Community...
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)

if %ERRORLEVEL% neq 0 (
    echo Error: Could not find Visual Studio 2022 environment
    pause
    exit /b 1
)

echo Visual Studio 2022 environment loaded successfully.

REM Verify cl.exe is available
where cl
if %ERRORLEVEL% neq 0 (
    echo Error: cl.exe not found in PATH
    pause
    exit /b 1
)

REM Set CUDA paths
set CUDA_INC_PATH=%CUDA_PATH%\include
set CUDA_LIB_PATH=%CUDA_PATH%\lib\x64

echo CUDA Include Path: %CUDA_INC_PATH%
echo CUDA Library Path: %CUDA_LIB_PATH%

REM Change to Native directory
cd MandelbrotMAUI\Native

echo Compiling CUDA wrapper DLL...
nvcc --shared -Xcompiler "/LD" -o MandelbrotCudaWrapper.dll MandelbrotCudaWrapper.cu MandelbrotCudaWrapper.def

if %ERRORLEVEL% neq 0 (
    echo Error during CUDA compilation
    cd ..\..
    pause
    exit /b 1
)

echo Copying DLL to root directory...
copy MandelbrotCudaWrapper.dll ..\..\
if %ERRORLEVEL% neq 0 (
    echo Warning: Could not copy DLL to root directory
)

cd ..\..

echo.
echo ===== BUILD SUCCESSFUL =====
echo DLL Location: MandelbrotMAUI\Native\MandelbrotCudaWrapper.dll
echo Also copied to: MandelbrotCudaWrapper.dll
echo.

echo Press any key to continue...
pause >nul