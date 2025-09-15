// Views/MandelbrotCanvas.cs - 繧ｫ繧ｹ繧ｿ繝謠冗判繧ｳ繝ｳ繝医Ο繝ｼ繝ｫ
using Microsoft.Maui.Graphics;
using MandelbrotMAUI.Services;
using MandelbrotMAUI.Models;

namespace MandelbrotMAUI.Views
{
    public class MandelbrotCanvas : GraphicsView, IDrawable
    {
        private readonly TileManager _tileManager;
        private ViewportState _viewport;
        private readonly Dictionary<TileKey, IImage> _renderedTiles;

        public MandelbrotCanvas(TileManager tileManager)
        {
            _tileManager = tileManager;
            _viewport = new ViewportState();
            _renderedTiles = new Dictionary<TileKey, IImage>();
            Drawable = this;
        }

        public void Draw(ICanvas canvas, RectF dirtyRect)
        {
            canvas.FillColor = Colors.Black;
            canvas.FillRectangle(dirtyRect);

            // 陦ｨ遉ｺ遽・峇縺ｮ繧ｿ繧､繝ｫ繧定ｨ育ｮ・
            var visibleTiles = CalculateVisibleTiles(dirtyRect);

            foreach (var tileInfo in visibleTiles)
            {
                DrawTile(canvas, tileInfo);
            }
        }

        private async void DrawTile(ICanvas canvas, TileInfo tileInfo)
        {
            var tileKey = new TileKey 
            { 
                X = tileInfo.X, 
                Y = tileInfo.Y, 
                ZoomLevel = GetZoomLevel(_viewport.Parameters.Zoom) 
            };

            // 繧ｭ繝｣繝・す繝･縺輔ｌ縺溽判蜒上ｒ繝√ぉ繝・け
            if (_renderedTiles.TryGetValue(tileKey, out var image))
            {
                canvas.DrawImage(image, 
                    tileInfo.ScreenX, tileInfo.ScreenY, 
                    tileInfo.Size, tileInfo.Size);
                return;
            }

            // 繧ｿ繧､繝ｫ繝・・繧ｿ繧帝撼蜷梧悄蜿門ｾ・
            _ = Task.Run(async () =>
            {
                try
                {
                    var imageData = await _tileManager.GetTileAsync(
                        _viewport.Parameters.CenterX,
                        _viewport.Parameters.CenterY,
                        _viewport.Parameters.Zoom,
                        tileInfo.X, tileInfo.Y,
                        _viewport.Parameters.MaxIterations);

                    // 繝舌う繝磯・蛻励ｒIImage縺ｫ螟画鋤
                    var stream = new MemoryStream(ConvertToRgbaStream(imageData));
                    var newImage = PlatformImage.FromStream(stream);
                    
                    _renderedTiles[tileKey] = newImage;
                    
                    // UI繧ｹ繝ｬ繝・ラ縺ｧ蜀肴緒逕ｻ
                    MainThread.BeginInvokeOnMainThread(() => Invalidate());
                }
                catch (Exception ex)
                {
                    // 繧ｨ繝ｩ繝ｼ繝上Φ繝峨Μ繝ｳ繧ｰ
                    System.Diagnostics.Debug.WriteLine($"Tile computation error: {ex.Message}");
                }
            });
        }

        // 繧ｸ繧ｧ繧ｹ繝√Ε繝ｼ繝上Φ繝峨Μ繝ｳ繧ｰ
        public void OnPanGesture(double deltaX, double deltaY)
        {
            double pixelSize = 1.0 / _viewport.Parameters.Zoom;
            _viewport.Parameters.CenterX -= deltaX * pixelSize;
            _viewport.Parameters.CenterY += deltaY * pixelSize;
            
            ClearOldTiles();
            Invalidate();
        }

        public void OnZoomGesture(double zoomFactor, double centerX, double centerY)
        {
            var (complexX, complexY) = _viewport.ScreenToComplex(centerX, centerY);
            
            _viewport.Parameters.Zoom *= zoomFactor;
            
            // 繧ｺ繝ｼ繝荳ｭ蠢・ｒ邯ｭ謖・
            _viewport.Parameters.CenterX = complexX;
            _viewport.Parameters.CenterY = complexY;
            
            ClearOldTiles();
            Invalidate();
        }

        private List<TileInfo> CalculateVisibleTiles(RectF viewport)
        {
            var tiles = new List<TileInfo>();
            int tileSize = 256;
            
            double pixelSize = 1.0 / _viewport.Parameters.Zoom;
            double tilesPerScreenX = viewport.Width / tileSize;
            double tilesPerScreenY = viewport.Height / tileSize;
            
            int startTileX = (int)Math.Floor(-tilesPerScreenX / 2) - 1;
            int endTileX = (int)Math.Ceiling(tilesPerScreenX / 2) + 1;
            int startTileY = (int)Math.Floor(-tilesPerScreenY / 2) - 1;
            int endTileY = (int)Math.Ceiling(tilesPerScreenY / 2) + 1;
            
            for (int tileY = startTileY; tileY <= endTileY; tileY++)
            {
                for (int tileX = startTileX; tileX <= endTileX; tileX++)
                {
                    double screenX = (tileX + tilesPerScreenX / 2) * tileSize;
                    double screenY = (tileY + tilesPerScreenY / 2) * tileSize;
                    
                    tiles.Add(new TileInfo
                    {
                        X = tileX,
                        Y = tileY,
                        ScreenX = screenX,
                        ScreenY = screenY,
                        Size = tileSize
                    });
                }
            }
            
            return tiles;
        }

        private void ClearOldTiles()
        {
            // 蜿､縺・ぜ繝ｼ繝繝ｬ繝吶Ν縺ｮ繧ｿ繧､繝ｫ繧偵け繝ｪ繧｢
            var currentZoomLevel = GetZoomLevel(_viewport.Parameters.Zoom);
            var keysToRemove = _renderedTiles.Keys
                .Where(k => Math.Abs(k.ZoomLevel - currentZoomLevel) > 2)
                .ToList();
                
            foreach (var key in keysToRemove)
            {
                _renderedTiles[key]?.Dispose();
                _renderedTiles.Remove(key);
            }
        }

        private int GetZoomLevel(double zoom) => (int)Math.Log2(Math.Max(1, zoom));

        private byte[] ConvertToRgbaStream(byte[] rgbaData)
        {
            // Platform-specific image format conversion
            // This would need platform-specific implementation
            return rgbaData;
        }
    }

    public class TileInfo
    {
        public int X { get; set; }
        public int Y { get; set; }
        public double ScreenX { get; set; }
        public double ScreenY { get; set; }
        public double Size { get; set; }
    }
}
