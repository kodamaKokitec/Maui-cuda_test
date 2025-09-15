using MandelbrotMAUI.Models;
using System.Runtime.InteropServices;

namespace MandelbrotMAUI.Services;

public class CudaMandelbrotService : IMandelbrotService
{
    [DllImport("MandelbrotCudaWrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int TestCudaOperation();

    [DllImport("MandelbrotCudaWrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int GenerateMandelbrot(
        byte[] imageData, int width, int height,
        double centerX, double centerY, double zoom, int maxIterations);

    [DllImport("MandelbrotCudaWrapper.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern int GetCudaDeviceInfo(
        byte[] deviceName, int nameSize, out int computeMajor, out int computeMinor);

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
        CudaDebugHelper.Log("Initializing CUDA...");
        try
        {
            if (TestCudaOperation() == 0)
            {
                _cudaAvailable = true;
                var deviceNameBuffer = new byte[256];
                int computeMajor, computeMinor;
                if (GetCudaDeviceInfo(deviceNameBuffer, deviceNameBuffer.Length, out computeMajor, out computeMinor) == 0)
                {
                    var deviceName = System.Text.Encoding.UTF8.GetString(deviceNameBuffer).TrimEnd('\0');
                    _gpuInfo = $"{deviceName} (Compute {computeMajor}.{computeMinor})";
                    CudaDebugHelper.Log($"CUDA initialized successfully: {_gpuInfo}");
                }
            }
            else
            {
                CudaDebugHelper.Log("CUDA test operation failed");
            }
        }
        catch (DllNotFoundException ex)
        {
            _cudaAvailable = false;
            CudaDebugHelper.Log($"CUDA DLL not found: {ex.Message}");
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"CUDA initialization failed: {ex.Message}");
            CudaDebugHelper.Log($"CUDA initialization failed: {ex.Message}");
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
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            CudaDebugHelper.Log($"Starting CUDA computation {width}x{height}, center=({centerX:F6}, {centerY:F6}), zoom={zoom:F2}");
            
            var rgbData = new byte[width * height * 3];
            int result = GenerateMandelbrot(rgbData, width, height, centerX, centerY, zoom, maxIterations);
            
            stopwatch.Stop();
            CudaDebugHelper.LogPerformance("Mandelbrot Generation", stopwatch.Elapsed, width * height);
            
            if (result != 0)
            {
                System.Diagnostics.Debug.WriteLine($"CUDA computation failed with error code: {result}");
                return _cpuFallback.ComputeTileAsync(centerX, centerY, zoom, width, height, maxIterations).Result;
            }
            
            var rgbaData = new byte[width * height * 4];
            for (int i = 0; i < width * height; i++)
            {
                int rgbIndex = i * 3;
                int rgbaIndex = i * 4;
                rgbaData[rgbaIndex] = rgbData[rgbIndex];
                rgbaData[rgbaIndex + 1] = rgbData[rgbIndex + 1];
                rgbaData[rgbaIndex + 2] = rgbData[rgbIndex + 2];
                rgbaData[rgbaIndex + 3] = 255;
            }
            
            return rgbaData;
        });
    }


}
