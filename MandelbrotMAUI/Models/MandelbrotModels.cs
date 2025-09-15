using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace MandelbrotMAUI.Models;

public class MandelbrotParameters : INotifyPropertyChanged
{
    private double _centerX = -0.5;
    private double _centerY = 0.0;
    private double _zoom = 1.0;
    private int _maxIterations = 1000;
    private int _tileSize = 256;

    public double CenterX
    {
        get => _centerX;
        set => SetProperty(ref _centerX, value);
    }

    public double CenterY
    {
        get => _centerY;
        set => SetProperty(ref _centerY, value);
    }

    public double Zoom
    {
        get => _zoom;
        set => SetProperty(ref _zoom, Math.Max(0.1, value));
    }

    public int MaxIterations
    {
        get => _maxIterations;
        set => SetProperty(ref _maxIterations, Math.Max(100, Math.Min(10000, value)));
    }

    public int TileSize
    {
        get => _tileSize;
        set => SetProperty(ref _tileSize, value);
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }

    protected bool SetProperty<T>(ref T backingStore, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(backingStore, value))
            return false;

        backingStore = value;
        OnPropertyChanged(propertyName);
        return true;
    }
}

public class ViewportState
{
    public double ViewportWidth { get; set; }
    public double ViewportHeight { get; set; }
    public MandelbrotParameters Parameters { get; set; } = new();

    public (double x, double y) ScreenToComplex(double screenX, double screenY)
    {
        double pixelSize = 1.0 / Parameters.Zoom;
        double complexX = Parameters.CenterX + (screenX - ViewportWidth / 2) * pixelSize;
        double complexY = Parameters.CenterY + (ViewportHeight / 2 - screenY) * pixelSize;
        return (complexX, complexY);
    }

    public (double x, double y) ComplexToScreen(double complexX, double complexY)
    {
        double pixelSize = 1.0 / Parameters.Zoom;
        double screenX = ViewportWidth / 2 + (complexX - Parameters.CenterX) / pixelSize;
        double screenY = ViewportHeight / 2 - (complexY - Parameters.CenterY) / pixelSize;
        return (screenX, screenY);
    }
}

public record TileKey(int X, int Y, int ZoomLevel);

public class TileData
{
    public byte[]? ImageData { get; set; }
    public DateTime LastAccessed { get; set; } = DateTime.Now;
    public bool IsComputing { get; set; } = false;
    public TaskCompletionSource<byte[]>? ComputationTask { get; set; }
}

public class TileInfo
{
    public int X { get; set; }
    public int Y { get; set; }
    public double ScreenX { get; set; }
    public double ScreenY { get; set; }
    public double Size { get; set; }
}

public interface IMandelbrotService
{
    bool IsAvailable { get; }
    string GetEngineInfo();
    Task<byte[]> ComputeTileAsync(double centerX, double centerY, double zoom, int width, int height, int maxIterations);
}
