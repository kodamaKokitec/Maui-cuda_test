@echo off
echo Setting up Visual Studio environment for x64...
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -arch=amd64

echo Compiling test_cuda_wrapper.cpp for x64...
cl /EHsc test_cuda_wrapper.cpp /Fe:test_cuda_wrapper_x64.exe

if %ERRORLEVEL% equ 0 (
    echo Build successful!
    echo You can now run: test_cuda_wrapper_x64.exe
) else (
    echo Build failed with error code %ERRORLEVEL%
)
pause
