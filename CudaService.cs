// Services/CudaService.cs
using System;
using System.Threading.Tasks;
using MandelbrotMAUI.Models;

namespace MandelbrotMAUI.Services
{
    public interface ICudaService
    {
        Task<byte[]> ComputeTileAsync(double centerX, double centerY, double zoom, 
                                     int width, int height, int maxIterations);
        string GetGpuInfo();
        bool IsAvailable { get; }
    }

    public class CudaService : ICudaService
    {
        public bool IsAvailable { get; private set; }

        public CudaService()
        {
            try
            {
                // CUDA 利用可能性チェック
                var info = GetGpuInfo();
                IsAvailable = !string.IsNullOrEmpty(info);
            }
            catch
            {
                IsAvailable = false;
            }
        }

        public async Task<byte[]> ComputeTileAsync(double centerX, double centerY, double zoom, 
                                                  int width, int height, int maxIterations)
        {
            if (!IsAvailable)
                throw new InvalidOperationException("CUDA is not available");

            return await Task.Run(() =>
            {
                // 高ズームレベルでダブル精度を使用
                if (zoom > 1000)
                {
                    return MandelbrotCudaEngine.ComputeTileDouble(
                        centerX, centerY, zoom, width, height, maxIterations);
                }
                else
                {
                    return MandelbrotCudaEngine.ComputeTile(
                        (float)centerX, (float)centerY, (float)zoom, 
                        width, height, maxIterations);
                }
            });
        }

        public string GetGpuInfo()
        {
            try
            {
                return MandelbrotCudaEngine.GetGpuInfo();
            }
            catch
            {
                return null;
            }
        }
    }

    // フォールバック用CPU実装
    public class CpuMandelbrotService : ICudaService
    {
        public bool IsAvailable => true;

        public async Task<byte[]> ComputeTileAsync(double centerX, double centerY, double zoom, 
                                                  int width, int height, int maxIterations)
        {
            return await Task.Run(() => ComputeTileCpu(centerX, centerY, zoom, width, height, maxIterations));
        }

        public string GetGpuInfo() => "CPU Fallback Mode";

        private byte[] ComputeTileCpu(double centerX, double centerY, double zoom, 
                                     int width, int height, int maxIterations)
        {
            byte[] imageData = new byte[width * height * 4]; // RGBA
            double pixelSize = 1.0 / zoom;

            Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    double complexX = centerX + (x - width / 2.0) * pixelSize;
                    double complexY = centerY + (height / 2.0 - y) * pixelSize;

                    int iter = ComputeMandelbrot(complexX, complexY, maxIterations);
                    int pixelIndex = (y * width + x) * 4;

                    if (iter == maxIterations)
                    {
                        // 黒
                        imageData[pixelIndex] = 0;     // R
                        imageData[pixelIndex + 1] = 0; // G
                        imageData[pixelIndex + 2] = 0; // B
                        imageData[pixelIndex + 3] = 255; // A
                    }
                    else
                    {
                        // カラフル
                        float ratio = (float)iter / maxIterations;
                        imageData[pixelIndex] = (byte)(255 * (1 - ratio));     // R
                        imageData[pixelIndex + 1] = (byte)(255 * ratio * 0.5); // G
                        imageData[pixelIndex + 2] = (byte)(255 * ratio);       // B
                        imageData[pixelIndex + 3] = 255; // A
                    }
                }
            });

            return imageData;
        }

        private int ComputeMandelbrot(double x, double y, int maxIterations)
        {
            double real = x, imag = y;
            int iter = 0;

            while (iter < maxIterations && (real * real + imag * imag) <= 4.0)
            {
                double temp = real * real - imag * imag + x;
                imag = 2.0 * real * imag + y;
                real = temp;
                iter++;
            }

            return iter;
        }
    }
}
