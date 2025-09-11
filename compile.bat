@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"
cd /d "c:\Users\kodama\OneDrive - アトミック株式会社\弘輝テック\cuda"
nvcc -o mandelbrot_cuda mandelbrot_cuda.cu
if %ERRORLEVEL% equ 0 (
    echo Build successful!
) else (
    echo Build failed with error code %ERRORLEVEL%
)
