namespace MandelbrotMAUI.Services;

public interface IMandelbrotService : IDisposable
{
    bool IsAvailable { get; }
    string GetEngineInfo();
    Task<byte[]> ComputeTileAsync(double centerX, double centerY, double zoom, int width, int height, int maxIterations);
}