using MandelbrotMAUI.Models;
using System.Runtime.InteropServices;

namespace MandelbrotMAUI.Services;

public class CudaMandelbrotService : IMandelbrotService
{
    // P/Invoke declarations for CUDA wrapper
    [DllImport("MandelbrotCudaWrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int GenerateMandelbrot(
        byte[] imageData, int width, int height,
        double centerX, double centerY, double zoom, int maxIterations);

    [DllImport("MandelbrotCudaWrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int GetCudaDeviceInfo(
        byte[] deviceName, int nameSize, out int computeMajor, out int computeMinor);

    [DllImport("MandelbrotCudaWrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int TestCudaOperation();

    private readonly CpuMandelbrotService _cpuFallback;
    private bool _cudaAvailable = false;
    private string _gpuInfo = "Unknown";

    public CudaMandelbrotService()
    {
        _cpuFallback = new CpuMandelbrotService();
        InitializeCudaIfAvailable();
    }

    public bool IsAvailable => _cudaAvailable || _cpuFallback.IsAvailable;

    public string GetEngineInfo() => _cudaAvailable ? $"CUDA GPU ({_gpuInfo})" : _cpuFallback.GetEngineInfo();

    private void InitializeCudaIfAvailable()
    {
        try
        {
            if (TestCudaOperation() == 0)
            {
                _cudaAvailable = true;

                // Get GPU info
                var deviceNameBuffer = new byte[256];
                int computeMajor, computeMinor;
                if (GetCudaDeviceInfo(deviceNameBuffer, deviceNameBuffer.Length, out computeMajor, out computeMinor) == 0)
                {
                    var deviceName = System.Text.Encoding.UTF8.GetString(deviceNameBuffer).TrimEnd('\0');
                    _gpuInfo = $"{deviceName} (Compute {computeMajor}.{computeMinor})";
                }
            }
        }
        catch (DllNotFoundException)
        {
            _cudaAvailable = false;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"CUDA initialization failed: {ex.Message}");
            _cudaAvailable = false;
        }
    }

    public async Task<byte[]> ComputeTileAsync(double centerX, double centerY, double zoom,
                                              int width, int height, int maxIterations)
    {
        if (_cudaAvailable)
        {
            return await ComputeTileWithCudaAsync(centerX, centerY, zoom, width, height, maxIterations);
        }
        else
        {
            return await _cpuFallback.ComputeTileAsync(centerX, centerY, zoom, width, height, maxIterations);
        }
    }

    private async Task<byte[]> ComputeTileWithCudaAsync(double centerX, double centerY, double zoom,
                                                       int width, int height, int maxIterations)
    {
        return await Task.Run(() =>
        {
            var rgbData = new byte[width * height * 3];
            
            int result = GenerateMandelbrot(rgbData, width, height, centerX, centerY, zoom, maxIterations);
            
            if (result != 0)
            {
                throw new InvalidOperationException($"CUDA computation failed with error code: {result}");
            }
            
            // Convert RGB to RGBA for UI compatibility
            var rgbaData = new byte[width * height * 4];
            for (int i = 0; i < width * height; i++)
            {
                int rgbIndex = i * 3;
                int rgbaIndex = i * 4;
                rgbaData[rgbaIndex] = rgbData[rgbIndex];         // R
                rgbaData[rgbaIndex + 1] = rgbData[rgbIndex + 1]; // G
                rgbaData[rgbaIndex + 2] = rgbData[rgbIndex + 2]; // B
                rgbaData[rgbaIndex + 3] = 255;                   // A (fully opaque)
            }
            
            return rgbaData;
        });
    }

    public void Dispose()
    {
        _cpuFallback?.Dispose();
    }
}