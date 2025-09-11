// Services/TileManager.cs
using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using MandelbrotMAUI.Models;

namespace MandelbrotMAUI.Services
{
    public class TileManager
    {
        private readonly ICudaService _cudaService;
        private readonly ConcurrentDictionary<TileKey, TileData> _tileCache;
        private readonly int _maxCacheSize = 1000;
        private readonly int _tileSize = 256;

        public TileManager(ICudaService cudaService)
        {
            _cudaService = cudaService;
            _tileCache = new ConcurrentDictionary<TileKey, TileData>();
        }

        public async Task<byte[]> GetTileAsync(double centerX, double centerY, double zoom, 
                                              int tileX, int tileY, int maxIterations)
        {
            var zoomLevel = GetZoomLevel(zoom);
            var tileKey = new TileKey { X = tileX, Y = tileY, ZoomLevel = zoomLevel };

            // キャッシュチェック
            if (_tileCache.TryGetValue(tileKey, out var cachedTile))
            {
                cachedTile.LastAccessed = DateTime.Now;
                return cachedTile.ImageData;
            }

            // 計算中チェック
            var computingTile = new TileData { IsComputing = true };
            if (!_tileCache.TryAdd(tileKey, computingTile))
            {
                // 他のスレッドが計算中
                while (_tileCache.TryGetValue(tileKey, out var tile) && tile.IsComputing)
                {
                    await Task.Delay(10);
                }
                return tile?.ImageData;
            }

            try
            {
                // タイル座標を複素平面座標に変換
                double pixelSize = 1.0 / zoom;
                double tileCenterX = centerX + (tileX - 0.5) * _tileSize * pixelSize;
                double tileCenterY = centerY + (0.5 - tileY) * _tileSize * pixelSize;

                // CUDA演算実行
                var imageData = await _cudaService.ComputeTileAsync(
                    tileCenterX, tileCenterY, zoom, _tileSize, _tileSize, maxIterations);

                // キャッシュに保存
                var newTile = new TileData 
                { 
                    ImageData = imageData, 
                    IsComputing = false,
                    LastAccessed = DateTime.Now 
                };
                _tileCache.TryUpdate(tileKey, newTile, computingTile);

                // キャッシュサイズ管理
                await ManageCacheSize();

                return imageData;
            }
            catch
            {
                _tileCache.TryRemove(tileKey, out _);
                throw;
            }
        }

        private int GetZoomLevel(double zoom)
        {
            // ズームレベルを段階的に分類
            return (int)Math.Log2(Math.Max(1, zoom));
        }

        private async Task ManageCacheSize()
        {
            if (_tileCache.Count <= _maxCacheSize) return;

            await Task.Run(() =>
            {
                var itemsToRemove = _tileCache.Count - _maxCacheSize + 100;
                var sortedItems = _tileCache
                    .Where(kvp => !kvp.Value.IsComputing)
                    .OrderBy(kvp => kvp.Value.LastAccessed)
                    .Take(itemsToRemove);

                foreach (var item in sortedItems)
                {
                    _tileCache.TryRemove(item.Key, out _);
                }
            });
        }

        public void ClearCache()
        {
            _tileCache.Clear();
        }

        public int CacheCount => _tileCache.Count;
    }
}
