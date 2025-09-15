@echo off
echo Setting up Visual Studio environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"

echo Compiling hello_cuda.cu...
nvcc -o hello_cuda.exe hello_cuda.cu

if %ERRORLEVEL% equ 0 (
    echo Build successful!
    echo You can now run: hello_cuda.exe
) else (
    echo Build failed with error code %ERRORLEVEL%
)
pause
