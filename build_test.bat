@echo off
echo Building test wrapper...

REM Set up Visual Studio 2022 environment
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"

if %ERRORLEVEL% neq 0 (
    echo Error: Could not find Visual Studio 2022 environment
    pause
    exit /b 1
)

echo Visual Studio environment loaded successfully.

REM Compile the test program
cl.exe test_rgb_values.cpp /Fe:test_rgb_values.exe

REM Compile the BMP test program  
cl.exe test_cuda_wrapper_bmp.cpp /Fe:test_cuda_wrapper_bmp.exe

echo Build completed!
pause
