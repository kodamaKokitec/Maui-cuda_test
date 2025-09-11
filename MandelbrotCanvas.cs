// Views/MandelbrotCanvas.cs - カスタム描画コントロール
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

            // 表示範囲のタイルを計算
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

            // キャッシュされた画像をチェック
            if (_renderedTiles.TryGetValue(tileKey, out var image))
            {
                canvas.DrawImage(image, 
                    tileInfo.ScreenX, tileInfo.ScreenY, 
                    tileInfo.Size, tileInfo.Size);
                return;
            }

            // タイルデータを非同期取得
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

                    // バイト配列をIImageに変換
                    var stream = new MemoryStream(ConvertToRgbaStream(imageData));
                    var newImage = PlatformImage.FromStream(stream);
                    
                    _renderedTiles[tileKey] = newImage;
                    
                    // UIスレッドで再描画
                    MainThread.BeginInvokeOnMainThread(() => Invalidate());
                }
                catch (Exception ex)
                {
                    // エラーハンドリング
                    System.Diagnostics.Debug.WriteLine($"Tile computation error: {ex.Message}");
                }
            });
        }

        // ジェスチャーハンドリング
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
            
            // ズーム中心を維持
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
            // 古いズームレベルのタイルをクリア
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
