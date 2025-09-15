@echo off
echo Setting up Visual Studio environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"

echo Compiling test_cuda_wrapper.cpp...
cl /EHsc test_cuda_wrapper.cpp /Fe:test_cuda_wrapper.exe

if %ERRORLEVEL% equ 0 (
    echo Build successful!
    echo You can now run: test_cuda_wrapper.exe
) else (
    echo Build failed with error code %ERRORLEVEL%
)
pause
