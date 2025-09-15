@echo off
echo CUDA Mandelbrot Set Visualization - Run Script
echo ===============================================

if not exist mandelbrot_cuda.exe (
    echo Error: mandelbrot_cuda.exe not found
    echo Please run build.bat first to compile the program
    pause
    exit /b 1
)

echo Running CUDA Mandelbrot visualization...
echo.

REM プログラムを実衁E
mandelbrot_cuda.exe

if %ERRORLEVEL% equ 0 (
    echo.
    echo Program completed successfully!
    if exist mandelbrot.ppm (
        echo Output file: mandelbrot.ppm
        echo.
        echo You can open mandelbrot.ppm with any image viewer
        echo Recommended viewers: IrfanView, GIMP, Photoshop
    )
) else (
    echo.
    echo Program execution failed!
    echo Error code: %ERRORLEVEL%
)

pause
