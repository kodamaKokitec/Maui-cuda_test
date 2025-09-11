@echo off
echo Building CUDA Mandelbrot Wrapper for MAUI...

REM Check if CUDA is available
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
if not exist "%CUDA_PATH%" (
    echo Error: CUDA not found at %CUDA_PATH%
    pause
    exit /b 1
)

REM Set up Visual Studio 2019 environment (fallback to VS2019)
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if %ERRORLEVEL% neq 0 (
    echo Trying VS2019 Enterprise...
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
)
if %ERRORLEVEL% neq 0 (
    echo Trying VS2019 Professional...
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
)
if %ERRORLEVEL% neq 0 (
    echo Trying VS2019 Community...
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
)

REM Set CUDA paths
set CUDA_INC_PATH=%CUDA_PATH%\include
set CUDA_LIB_PATH=%CUDA_PATH%\lib\x64

echo CUDA Include Path: %CUDA_INC_PATH%
echo CUDA Library Path: %CUDA_LIB_PATH%

REM Change to Native directory
cd /d "%~dp0MandelbrotMAUI\Native"

echo Building CUDA wrapper DLL...

REM Compile CUDA source to object file
nvcc -c -o MandelbrotCudaWrapper.obj MandelbrotCudaWrapper.cu ^
    -I"%CUDA_INC_PATH%" ^
    --compiler-options "/MD /EHsc" ^
    -arch=sm_75 ^
    --ptxas-options=-v

if %ERRORLEVEL% neq 0 (
    echo Error: CUDA compilation failed
    pause
    exit /b 1
)

REM Link to create DLL
link /DLL /OUT:MandelbrotCudaWrapper.dll ^
    MandelbrotCudaWrapper.obj ^
    cudart.lib ^
    /LIBPATH:"%CUDA_LIB_PATH%" ^
    /MACHINE:X64 ^
    /SUBSYSTEM:WINDOWS

if %ERRORLEVEL% neq 0 (
    echo Error: DLL linking failed
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo DLL: %CD%\MandelbrotCudaWrapper.dll
echo.

REM Clean up object files
del *.obj *.exp *.lib 2>nul

echo Build process finished.
pause
