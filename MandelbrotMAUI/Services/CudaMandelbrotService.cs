using MandelbrotMAUI.Models;
using System.Runtime.InteropServices;

namespace MandelbrotMAUI.Services;

public class CudaMandelbrotService : IMandelbrotService
{
    // P/Invoke declarations for CUDA wrapper
    [DllImport("MandelbrotCudaWrapper.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "InitializeCuda")]
    private static extern int InitializeCuda();

    [DllImport("MandelbrotCudaWrapper.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "CalculateMandelbrotCuda")]
    private static extern int CalculateMandelbrotCuda(
        byte[] outputBuffer, int width, int height,
        double centerX, double centerY, double zoom, int maxIterations);

    [DllImport("MandelbrotCudaWrapper.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "IsCudaAvailable")]
    private static extern int IsCudaAvailable();

    [DllImport("MandelbrotCudaWrapper.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetGpuInfo")]
    private static extern void GetGpuInfo(byte[] infoBuffer, int bufferSize);

    [DllImport("MandelbrotCudaWrapper.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "CleanupCuda")]
    private static extern void CleanupCuda();

    private readonly CpuMandelbrotService _cpuFallback;
    private bool _cudaInitialized = false;
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
            if (IsCudaAvailable() > 0)
            {
                if (InitializeCuda() == 0)
                {
                    _cudaAvailable = true;
                    _cudaInitialized = true;

                    // Get GPU info
                    var infoBuffer = new byte[256];
                    GetGpuInfo(infoBuffer, infoBuffer.Length);
                    _gpuInfo = System.Text.Encoding.UTF8.GetString(infoBuffer).TrimEnd('\0');
                }
            }
        }
        catch (DllNotFoundException)
        {
            // CUDA wrapper DLL not found - use CPU fallback
            _cudaAvailable = false;
        }
        catch (Exception)
        {
            // Other initialization errors - use CPU fallback
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
            var rgbData = new byte[width * height * 3]; // RGB
            
            System.Diagnostics.Debug.WriteLine($"CUDA: Computing tile {width}x{height}, center=({centerX:F6}, {centerY:F6}), zoom={zoom:F2}");
            
            int result = CalculateMandelbrotCuda(rgbData, width, height, centerX, centerY, zoom, maxIterations);
            
            System.Diagnostics.Debug.WriteLine($"CUDA: Computation result = {result}");
            
            if (result != 0)
            {
                System.Diagnostics.Debug.WriteLine($"CUDA: Failed with result {result}, falling back to CPU");
                // CUDA calculation failed, fallback to CPU
                return _cpuFallback.ComputeTileAsync(centerX, centerY, zoom, width, height, maxIterations).Result;
            }

            // RGB データのサンプリングを確認
            if (rgbData.Length >= 12)
            {
                System.Diagnostics.Debug.WriteLine($"CUDA RGB samples: [{rgbData[0]},{rgbData[1]},{rgbData[2]}] [{rgbData[3]},{rgbData[4]},{rgbData[5]}] [{rgbData[6]},{rgbData[7]},{rgbData[8]}] [{rgbData[9]},{rgbData[10]},{rgbData[11]}]");
            }

            // Convert RGB to RGBA
            var rgbaData = new byte[width * height * 4];
            for (int i = 0; i < width * height; i++)
            {
                rgbaData[i * 4] = rgbData[i * 3];     // R
                rgbaData[i * 4 + 1] = rgbData[i * 3 + 1]; // G
                rgbaData[i * 4 + 2] = rgbData[i * 3 + 2]; // B
                rgbaData[i * 4 + 3] = 255;            // A
            }

            // RGBA データのサンプリングを確認
            if (rgbaData.Length >= 16)
            {
                System.Diagnostics.Debug.WriteLine($"CUDA RGBA samples: [{rgbaData[0]},{rgbaData[1]},{rgbaData[2]},{rgbaData[3]}] [{rgbaData[4]},{rgbaData[5]},{rgbaData[6]},{rgbaData[7]}] [{rgbaData[8]},{rgbaData[9]},{rgbaData[10]},{rgbaData[11]}] [{rgbaData[12]},{rgbaData[13]},{rgbaData[14]},{rgbaData[15]}]");
            }

            // デバッグ用：最初のタイル計算時にBMPファイルを保存
            SaveDebugBmp(rgbData, width, height, centerX, centerY, zoom);

            return rgbaData;
        });
    }

    private static int _debugSaveCount = 0;

    private void SaveDebugBmp(byte[] rgbData, int width, int height, double centerX, double centerY, double zoom)
    {
        // 最初の数枚のタイルのみ保存（ファイル数制限）
        if (_debugSaveCount >= 3) return;
        
        _debugSaveCount++;

        try
        {
            string desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            string fileName = Path.Combine(desktopPath, $"debug_tile_{_debugSaveCount}_{width}x{height}_{centerX:F3}_{centerY:F3}_z{zoom:F1}.bmp");
            
            SaveBmpFile(fileName, rgbData, width, height);
            System.Diagnostics.Debug.WriteLine($"Debug BMP saved: {fileName}");
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error saving debug BMP: {ex.Message}");
        }
    }

    private void SaveBmpFile(string fileName, byte[] rgbData, int width, int height)
    {
        // BMP file format
        int imageSize = width * height * 3;
        int fileSize = 54 + imageSize; // BMP header is 54 bytes
        
        using (var fs = new FileStream(fileName, FileMode.Create))
        using (var writer = new BinaryWriter(fs))
        {
            // BMP file header (14 bytes)
            writer.Write((byte)'B');
            writer.Write((byte)'M');
            writer.Write(fileSize);        // File size
            writer.Write((int)0);          // Reserved
            writer.Write(54);              // Offset to image data
            
            // BMP info header (40 bytes)
            writer.Write(40);              // Info header size
            writer.Write(width);           // Image width
            writer.Write(height);          // Image height
            writer.Write((short)1);        // Planes
            writer.Write((short)24);       // Bits per pixel
            writer.Write(0);               // Compression
            writer.Write(imageSize);       // Image size
            writer.Write(0);               // X pixels per meter
            writer.Write(0);               // Y pixels per meter
            writer.Write(0);               // Colors used
            writer.Write(0);               // Important colors
            
            // BMP data is stored bottom-to-top, so we need to flip the image
            for (int y = height - 1; y >= 0; y--)
            {
                for (int x = 0; x < width; x++)
                {
                    int srcIndex = (y * width + x) * 3;
                    // BMP uses BGR order, while our data is RGB
                    writer.Write(rgbData[srcIndex + 2]); // B
                    writer.Write(rgbData[srcIndex + 1]); // G
                    writer.Write(rgbData[srcIndex]);     // R
                }
            }
        }
    }

    ~CudaMandelbrotService()
    {
        if (_cudaInitialized)
        {
            CleanupCuda();
        }
    }
}
