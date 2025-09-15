<<<<<<< HEAD
using MandelbrotMAUI.Models;

namespace MandelbrotMAUI.Services;

public interface IMandelbrotService
{
    Task<byte[]> ComputeTileAsync(double centerX, double centerY, double zoom, 
                                 int width, int height, int maxIterations);
    string GetEngineInfo();
    bool IsAvailable { get; }
}

public class CpuMandelbrotService : IMandelbrotService
{
    public bool IsAvailable => true;

    public async Task<byte[]> ComputeTileAsync(double centerX, double centerY, double zoom, 
                                              int width, int height, int maxIterations)
    {
        return await Task.Run(() => ComputeTileCpu(centerX, centerY, zoom, width, height, maxIterations));
    }

    public string GetEngineInfo() => "CPU Implementation (Parallel Processing)";

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
                    // Inside set - black
                    imageData[pixelIndex] = 0;     // R
                    imageData[pixelIndex + 1] = 0; // G
                    imageData[pixelIndex + 2] = 0; // B
                    imageData[pixelIndex + 3] = 255; // A
                }
                else
                {
                    // Outside set - colorful HSV-based coloring
                    var (r, g, b) = HsvToRgb(iter, maxIterations);
                    imageData[pixelIndex] = r;     // R
                    imageData[pixelIndex + 1] = g; // G
                    imageData[pixelIndex + 2] = b; // B
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

    private (byte r, byte g, byte b) HsvToRgb(int iteration, int maxIterations)
    {
        if (iteration == maxIterations)
            return (0, 0, 0);

        float hue = ((float)iteration / maxIterations) * 360.0f;
        float saturation = 1.0f;
        float value = 1.0f;

        float c = value * saturation;
        float x = c * (1 - Math.Abs((hue / 60.0f) % 2 - 1));
        float m = value - c;

        float r, g, b;
        if (hue < 60) { r = c; g = x; b = 0; }
        else if (hue < 120) { r = x; g = c; b = 0; }
        else if (hue < 180) { r = 0; g = c; b = x; }
        else if (hue < 240) { r = 0; g = x; b = c; }
        else if (hue < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }

        return ((byte)((r + m) * 255), (byte)((g + m) * 255), (byte)((b + m) * 255));
    }
}
=======
namespace MandelbrotMAUI.Services;

public interface IMandelbrotService : IDisposable
{
    bool IsAvailable { get; }
    string GetEngineInfo();
    Task<byte[]> ComputeTileAsync(double centerX, double centerY, double zoom, int width, int height, int maxIterations);
}
>>>>>>> 714a192637bdc28463b85e4fc8f387b4f517cf83
