// Models/MandelbrotParameters.cs
using System;

namespace MandelbrotMAUI.Models
{
    public class MandelbrotParameters
    {
        public double CenterX { get; set; } = -0.5;
        public double CenterY { get; set; } = 0.0;
        public double Zoom { get; set; } = 1.0;
        public int MaxIterations { get; set; } = 1000;
        public int TileSize { get; set; } = 256;
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
    }

    public class TileKey
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int ZoomLevel { get; set; }
        
        public override bool Equals(object obj) =>
            obj is TileKey key && X == key.X && Y == key.Y && ZoomLevel == key.ZoomLevel;
            
        public override int GetHashCode() => HashCode.Combine(X, Y, ZoomLevel);
    }

    public class TileData
    {
        public byte[] ImageData { get; set; }
        public DateTime LastAccessed { get; set; } = DateTime.Now;
        public bool IsComputing { get; set; } = false;
    }
}
