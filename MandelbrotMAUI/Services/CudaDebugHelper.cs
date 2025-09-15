using System.Diagnostics;
using System.Runtime.InteropServices;

namespace MandelbrotMAUI.Services;

/// <summary>
/// Debug helper for Visual Studio debugging of CUDA operations
/// </summary>
public static class CudaDebugHelper
{
    [DllImport("kernel32.dll")]
    private static extern bool AllocConsole();

    [DllImport("kernel32.dll")]
    private static extern bool FreeConsole();

    private static bool _consoleAllocated = false;

    /// <summary>
    /// Allocate console for debug output during Visual Studio debugging
    /// </summary>
    public static void EnableConsoleOutput()
    {
#if DEBUG
        if (!_consoleAllocated)
        {
            AllocConsole();
            _consoleAllocated = true;
            Console.WriteLine("CUDA Debug Console Enabled");
        }
#endif
    }

    /// <summary>
    /// Log debug information with timestamp
    /// </summary>
    public static void Log(string message)
    {
#if DEBUG
        var timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
        var output = $"[{timestamp}] {message}";
        
        Debug.WriteLine(output);
        Console.WriteLine(output);
        
        // Also write to debug file for persistent logging
        try
        {
            var logPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "cuda_debug.log");
            File.AppendAllText(logPath, output + Environment.NewLine);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Failed to write debug log: {ex.Message}");
        }
#endif
    }

    /// <summary>
    /// Log CUDA operation performance metrics
    /// </summary>
    public static void LogPerformance(string operation, TimeSpan elapsed, int pixelCount = 0)
    {
#if DEBUG
        var message = $"CUDA {operation}: {elapsed.TotalMilliseconds:F2}ms";
        if (pixelCount > 0)
        {
            var pixelsPerSecond = pixelCount / elapsed.TotalSeconds;
            message += $" ({pixelsPerSecond / 1000000:F2} Mpixels/sec)";
        }
        Log(message);
#endif
    }

    /// <summary>
    /// Free console when application exits
    /// </summary>
    public static void Cleanup()
    {
#if DEBUG
        if (_consoleAllocated)
        {
            Console.WriteLine("CUDA Debug Console Cleanup");
            FreeConsole();
            _consoleAllocated = false;
        }
#endif
    }
}