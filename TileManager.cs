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

            // 繧ｭ繝｣繝・す繝･繝√ぉ繝・け
            if (_tileCache.TryGetValue(tileKey, out var cachedTile))
            {
                cachedTile.LastAccessed = DateTime.Now;
                return cachedTile.ImageData;
            }

            // 險育ｮ嶺ｸｭ繝√ぉ繝・け
            var computingTile = new TileData { IsComputing = true };
            if (!_tileCache.TryAdd(tileKey, computingTile))
            {
                // 莉悶・繧ｹ繝ｬ繝・ラ縺瑚ｨ育ｮ嶺ｸｭ
                while (_tileCache.TryGetValue(tileKey, out var tile) && tile.IsComputing)
                {
                    await Task.Delay(10);
                }
                return tile?.ImageData;
            }

            try
            {
                // 繧ｿ繧､繝ｫ蠎ｧ讓吶ｒ隍・ｴ蟷ｳ髱｢蠎ｧ讓吶↓螟画鋤
                double pixelSize = 1.0 / zoom;
                double tileCenterX = centerX + (tileX - 0.5) * _tileSize * pixelSize;
                double tileCenterY = centerY + (0.5 - tileY) * _tileSize * pixelSize;

                // CUDA貍皮ｮ怜ｮ溯｡・
                var imageData = await _cudaService.ComputeTileAsync(
                    tileCenterX, tileCenterY, zoom, _tileSize, _tileSize, maxIterations);

                // 繧ｭ繝｣繝・す繝･縺ｫ菫晏ｭ・
                var newTile = new TileData 
                { 
                    ImageData = imageData, 
                    IsComputing = false,
                    LastAccessed = DateTime.Now 
                };
                _tileCache.TryUpdate(tileKey, newTile, computingTile);

                // 繧ｭ繝｣繝・す繝･繧ｵ繧､繧ｺ邂｡逅・
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
            // 繧ｺ繝ｼ繝繝ｬ繝吶Ν繧呈ｮｵ髫守噪縺ｫ蛻・｡・
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
