@echo off
echo Building CUDA Mandelbrot Wrapper for MAUI...

REM Check if CUDA is available
if not exist "%CUDA_PATH%" (
    echo Error: CUDA not found. Please ensure CUDA is installed and CUDA_PATH is set.
    echo Current CUDA_PATH: %CUDA_PATH%
    pause
    exit /b 1
)

REM Set up Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

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

REM Copy DLL to output directory
copy MandelbrotCudaWrapper.dll ..\bin\Debug\net10.0-windows10.0.19041.0\win-x64\
copy MandelbrotCudaWrapper.dll ..\bin\Release\net10.0-windows10.0.19041.0\win-x64\

echo.
echo Build completed successfully!
echo DLL: %CD%\MandelbrotCudaWrapper.dll
echo.

REM Clean up object files
del *.obj *.exp *.lib 2>nul

echo Build process finished.
pause
