using System.Collections.Concurrent;
using MandelbrotMAUI.Models;

namespace MandelbrotMAUI.Services;

public class TileManager
{
    private readonly IMandelbrotService _mandelbrotService;
    private readonly ConcurrentDictionary<TileKey, TileData> _tileCache;
    private readonly int _maxCacheSize = 1000;
    private readonly int _tileSize = 256;

    public int TileSize => _tileSize;

    public TileManager(IMandelbrotService mandelbrotService)
    {
        _mandelbrotService = mandelbrotService;
        _tileCache = new ConcurrentDictionary<TileKey, TileData>();
    }

    public async Task<byte[]> GetTileAsync(double centerX, double centerY, double zoom, 
                                          int tileX, int tileY, int maxIterations)
    {
        var zoomLevel = GetZoomLevel(zoom);
        var tileKey = new TileKey(tileX, tileY, zoomLevel);

        // キャッシュチェック
        if (_tileCache.TryGetValue(tileKey, out var cachedTile))
        {
            cachedTile.LastAccessed = DateTime.Now;
            
            if (cachedTile.ImageData != null)
                return cachedTile.ImageData;
            
            // 計算中の場合は完了を待つ
            if (cachedTile.ComputationTask != null)
                return await cachedTile.ComputationTask.Task;
        }

        // 新しい計算タスクを開始
        var tcs = new TaskCompletionSource<byte[]>();
        var newTile = new TileData 
        { 
            IsComputing = true, 
            ComputationTask = tcs,
            LastAccessed = DateTime.Now 
        };

        if (!_tileCache.TryAdd(tileKey, newTile))
        {
            // 他のスレッドが既に開始している
            if (_tileCache.TryGetValue(tileKey, out var existingTile) && 
                existingTile.ComputationTask != null)
            {
                return await existingTile.ComputationTask.Task;
            }
        }

        try
        {
            // タイル座標を複素平面座標に変換
            double pixelSize = 1.0 / zoom;
            double tileCenterX = centerX + (tileX - 0.5) * _tileSize * pixelSize;
            double tileCenterY = centerY + (0.5 - tileY) * _tileSize * pixelSize;

            // 演算実行
            var imageData = await _mandelbrotService.ComputeTileAsync(
                tileCenterX, tileCenterY, zoom, _tileSize, _tileSize, maxIterations);

            // キャッシュに保存
            newTile.ImageData = imageData;
            newTile.IsComputing = false;
            tcs.SetResult(imageData);

            // キャッシュサイズ管理
            _ = Task.Run(ManageCacheSize);

            return imageData;
        }
        catch (Exception ex)
        {
            _tileCache.TryRemove(tileKey, out _);
            tcs.SetException(ex);
            throw;
        }
    }

    private int GetZoomLevel(double zoom)
    {
        // ズームレベルを段階的に分類 (2の累乗ベース)
        return (int)Math.Floor(Math.Log2(Math.Max(1, zoom)));
    }

    private async Task ManageCacheSize()
    {
        if (_tileCache.Count <= _maxCacheSize) return;

        await Task.Run(() =>
        {
            var itemsToRemove = _tileCache.Count - _maxCacheSize + 100;
            var sortedItems = _tileCache
                .Where(kvp => !kvp.Value.IsComputing && kvp.Value.ImageData != null)
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

    public List<TileInfo> CalculateVisibleTiles(double viewportWidth, double viewportHeight, 
                                               ViewportState viewport)
    {
        var tiles = new List<TileInfo>();
        
        double pixelSize = 1.0 / viewport.Parameters.Zoom;
        double tilesPerScreenX = viewportWidth / _tileSize;
        double tilesPerScreenY = viewportHeight / _tileSize;
        
        int startTileX = (int)Math.Floor(-tilesPerScreenX / 2) - 1;
        int endTileX = (int)Math.Ceiling(tilesPerScreenX / 2) + 1;
        int startTileY = (int)Math.Floor(-tilesPerScreenY / 2) - 1;
        int endTileY = (int)Math.Ceiling(tilesPerScreenY / 2) + 1;
        
        for (int tileY = startTileY; tileY <= endTileY; tileY++)
        {
            for (int tileX = startTileX; tileX <= endTileX; tileX++)
            {
                double screenX = (tileX + tilesPerScreenX / 2) * _tileSize;
                double screenY = (tileY + tilesPerScreenY / 2) * _tileSize;
                
                tiles.Add(new TileInfo
                {
                    X = tileX,
                    Y = tileY,
                    ScreenX = screenX,
                    ScreenY = screenY,
                    Size = _tileSize
                });
            }
        }
        
        return tiles;
    }
}
