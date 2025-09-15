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
<<<<<<< HEAD
cd /d "%~dp0MandelbrotMAUI\Native"

echo Building CUDA wrapper DLL...

REM Compile CUDA source to object file
nvcc -c -o MandelbrotCudaWrapper.obj MandelbrotCudaWrapper.cu ^
    -I"%CUDA_INC_PATH%" ^
    --compiler-options "/MD /EHsc /W3" ^
    -arch=sm_75 ^
    --ptxas-options=-v ^
    -O2

if %ERRORLEVEL% neq 0 (
    echo Error: CUDA compilation failed
=======
cd MandelbrotMAUI\Native

echo Compiling CUDA wrapper DLL...
nvcc --shared -Xcompiler "/LD" -o MandelbrotCudaWrapper.dll MandelbrotCudaWrapper.cu MandelbrotCudaWrapper.def

if %ERRORLEVEL% neq 0 (
    echo Error during CUDA compilation
    cd ..\..
>>>>>>> 714a192637bdc28463b85e4fc8f387b4f517cf83
    pause
    exit /b 1
)

<<<<<<< HEAD
echo CUDA compilation successful!

REM Link to create DLL
link /DLL /OUT:MandelbrotCudaWrapper.dll ^
    MandelbrotCudaWrapper.obj ^
    cudart.lib ^
    /LIBPATH:"%CUDA_LIB_PATH%" ^
    /MACHINE:X64 ^
    /SUBSYSTEM:WINDOWS ^
    /DEF:MandelbrotCudaWrapper.def ^
    /NOLOGO

if %ERRORLEVEL% neq 0 (
    echo Error: DLL linking failed
    pause
    exit /b 1
)

echo DLL linking successful!

REM Create output directories if they don't exist
if not exist "..\bin\Debug\net10.0-windows10.0.19041.0\win-x64\" mkdir "..\bin\Debug\net10.0-windows10.0.19041.0\win-x64\"
if not exist "..\bin\Release\net10.0-windows10.0.19041.0\win-x64\" mkdir "..\bin\Release\net10.0-windows10.0.19041.0\win-x64\"

REM Copy DLL to output directories
copy MandelbrotCudaWrapper.dll "..\bin\Debug\net10.0-windows10.0.19041.0\win-x64\" >nul
copy MandelbrotCudaWrapper.dll "..\bin\Release\net10.0-windows10.0.19041.0\win-x64\" >nul

echo.
echo Build completed successfully!
echo DLL: %CD%\MandelbrotCudaWrapper.dll
echo Copied to MAUI output directories.
echo.

REM Clean up object files
del *.obj *.exp *.lib 2>nul

echo Build process finished.
pause
=======
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
>>>>>>> 714a192637bdc28463b85e4fc8f387b4f517cf83
