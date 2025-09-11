using Microsoft.Maui.Graphics;
using MandelbrotMAUI.Models;
using MandelbrotMAUI.Services;
using System.Collections.Concurrent;
using IImage = Microsoft.Maui.Graphics.IImage;

#if WINDOWS
using Microsoft.Maui.Graphics.Win2D;
#endif

namespace MandelbrotMAUI.Views;

public class TileRenderData
{
    public int X { get; set; }
    public int Y { get; set; }
    public int Size { get; set; }
    public byte[] RgbaData { get; set; } = Array.Empty<byte>();
    public double ScreenX { get; set; }
    public double ScreenY { get; set; }
}

public class MandelbrotCanvas : GraphicsView, IDrawable
{
    private readonly TileManager _tileManager;
    private ViewportState _viewport;
    private readonly ConcurrentDictionary<TileKey, IImage> _renderedTiles;
    private readonly ConcurrentDictionary<TileKey, TileRenderData> _renderedTileData;
    private bool _isDragging = false;
    private Point _lastPanPoint;

    public MandelbrotCanvas(TileManager tileManager)
    {
        _tileManager = tileManager;
        _viewport = new ViewportState();
        _renderedTiles = new ConcurrentDictionary<TileKey, IImage>();
        _renderedTileData = new ConcurrentDictionary<TileKey, TileRenderData>();
        Drawable = this;
        
        // ジェスチャー認識の設定
        var panGesture = new PanGestureRecognizer();
        panGesture.PanUpdated += OnPanUpdated;
        GestureRecognizers.Add(panGesture);

        var pinchGesture = new PinchGestureRecognizer();
        pinchGesture.PinchUpdated += OnPinchUpdated;
        GestureRecognizers.Add(pinchGesture);

        var tapGesture = new TapGestureRecognizer();
        tapGesture.Tapped += OnTapped;
        GestureRecognizers.Add(tapGesture);
    }

    public ViewportState Viewport => _viewport;

    public void Draw(ICanvas canvas, RectF dirtyRect)
    {
        // 背景を黒で塗りつぶし
        canvas.FillColor = Colors.Black;
        canvas.FillRectangle(dirtyRect);

        // ビューポートサイズを更新
        _viewport.ViewportWidth = dirtyRect.Width;
        _viewport.ViewportHeight = dirtyRect.Height;

        // 表示範囲のタイルを計算
        var visibleTiles = _tileManager.CalculateVisibleTiles(
            dirtyRect.Width, dirtyRect.Height, _viewport);

        // 各タイルを描画
        foreach (var tileInfo in visibleTiles)
        {
            DrawTile(canvas, tileInfo);
        }

        // デバッグ情報を表示
        DrawDebugInfo(canvas, dirtyRect);
    }

    private void DrawTile(ICanvas canvas, TileInfo tileInfo)
    {
        var zoomLevel = GetZoomLevel(_viewport.Parameters.Zoom);
        var tileKey = new TileKey(tileInfo.X, tileInfo.Y, zoomLevel);

        // RGBAデータによる直接描画をチェック
        if (_renderedTileData.TryGetValue(tileKey, out var tileData))
        {
            DrawTileFromRgbaData(canvas, tileData);
            return;
        }

        // キャッシュされた画像をチェック
        if (_renderedTiles.TryGetValue(tileKey, out var image))
        {
            canvas.DrawImage(image, 
                (float)tileInfo.ScreenX, (float)tileInfo.ScreenY, 
                (float)tileInfo.Size, (float)tileInfo.Size);
            return;
        }

        // プレースホルダーを描画
        canvas.StrokeColor = Colors.Gray;
        canvas.StrokeSize = 1;
        canvas.DrawRectangle((float)tileInfo.ScreenX, (float)tileInfo.ScreenY, 
                           (float)tileInfo.Size, (float)tileInfo.Size);

        // 中央に "Computing..." テキストを表示
        canvas.FontColor = Colors.White;
        canvas.FontSize = 10;
        canvas.DrawString("Computing...", 
            (float)tileInfo.ScreenX + 10, (float)tileInfo.ScreenY + 10, 
            100, 20, HorizontalAlignment.Left, VerticalAlignment.Top);

        // タイルデータを非同期取得
        _ = Task.Run(async () =>
        {
            try
            {
                System.Diagnostics.Debug.WriteLine($"Starting tile computation for ({tileInfo.X}, {tileInfo.Y})");
                
                var imageData = await _tileManager.GetTileAsync(
                    _viewport.Parameters.CenterX,
                    _viewport.Parameters.CenterY,
                    _viewport.Parameters.Zoom,
                    tileInfo.X, tileInfo.Y,
                    _viewport.Parameters.MaxIterations);

                System.Diagnostics.Debug.WriteLine($"Tile computation completed for ({tileInfo.X}, {tileInfo.Y}), data length: {imageData?.Length ?? 0}");

                // CUDA実装: RGBAデータを直接描画用に保存
                if (imageData != null && imageData.Length > 0)
                {
                    System.Diagnostics.Debug.WriteLine($"Creating tile data for ({tileInfo.X}, {tileInfo.Y})");
                    var tileSize = _tileManager.TileSize;
                    
                    // データサンプルを詳しく確認
                    if (imageData.Length >= 64)
                    {
                        System.Diagnostics.Debug.WriteLine($"RGBA data analysis for tile ({tileInfo.X}, {tileInfo.Y}):");
                        System.Diagnostics.Debug.WriteLine($"Total length: {imageData.Length} bytes");
                        System.Diagnostics.Debug.WriteLine($"Expected size: {tileSize}x{tileSize} = {tileSize * tileSize * 4} bytes");
                        
                        // 複数のサンプルピクセルを確認
                        for (int i = 0; i < Math.Min(16, imageData.Length / 4); i++)
                        {
                            var pixelIndex = i * 4;
                            var r = imageData[pixelIndex];
                            var g = imageData[pixelIndex + 1];
                            var b = imageData[pixelIndex + 2];
                            var a = imageData[pixelIndex + 3];
                            System.Diagnostics.Debug.WriteLine($"Pixel {i}: R={r}, G={g}, B={b}, A={a}");
                        }
                        
                        // 色の分布を確認
                        int redCount = 0, greenCount = 0, blueCount = 0, blackCount = 0;
                        for (int i = 0; i < imageData.Length; i += 4)
                        {
                            var r = imageData[i];
                            var g = imageData[i + 1];
                            var b = imageData[i + 2];
                            
                            if (r > g && r > b && r > 50) redCount++;
                            else if (g > r && g > b && g > 50) greenCount++;
                            else if (b > r && b > g && b > 50) blueCount++;
                            else if (r < 50 && g < 50 && b < 50) blackCount++;
                        }
                        
                        var totalPixels = imageData.Length / 4;
                        System.Diagnostics.Debug.WriteLine($"Color distribution - Red: {redCount}/{totalPixels}, Green: {greenCount}/{totalPixels}, Blue: {blueCount}/{totalPixels}, Black: {blackCount}/{totalPixels}");
                    }
                    
                    // RGBAデータをタイル情報として保存
                    var tileRenderData = new TileRenderData
                    {
                        X = tileInfo.X,
                        Y = tileInfo.Y,
                        Size = tileSize,
                        RgbaData = imageData,
                        ScreenX = tileInfo.ScreenX,
                        ScreenY = tileInfo.ScreenY
                    };
                    
                    // キャッシュに追加（IImageの代わりにRGBAデータを保存）
                    _renderedTileData.TryAdd(tileKey, tileRenderData);
                    System.Diagnostics.Debug.WriteLine($"Tile data cached for ({tileInfo.X}, {tileInfo.Y})");
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine($"No image data received for tile ({tileInfo.X}, {tileInfo.Y})");
                }
                
                // UIスレッドで再描画
                System.Diagnostics.Debug.WriteLine($"Requesting UI invalidation for tile ({tileInfo.X}, {tileInfo.Y})");
                MainThread.BeginInvokeOnMainThread(() => Invalidate());
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Tile computation error for ({tileInfo.X}, {tileInfo.Y}): {ex.Message}");
                System.Diagnostics.Debug.WriteLine($"Stack trace: {ex.StackTrace}");
                
                try
                {
                    var logFile = Path.Combine(AppContext.BaseDirectory, "canvas_debug.log");
                    File.AppendAllText(logFile, $"{DateTime.Now:HH:mm:ss.fff} - Tile computation error for ({tileInfo.X}, {tileInfo.Y}): {ex.Message}\n");
                    File.AppendAllText(logFile, $"{DateTime.Now:HH:mm:ss.fff} - Stack trace: {ex.StackTrace}\n");
                }
                catch { /* ログファイル書き込み失敗は無視 */ }
            }
        });
    }

    private void DrawTileFromRgbaData(ICanvas canvas, TileRenderData tileData)
    {
        try
        {
            // RGBAデータを使って直接ピクセルを描画
            var rgbaData = tileData.RgbaData;
            var size = tileData.Size;
            var startX = (float)tileData.ScreenX;
            var startY = (float)tileData.ScreenY;
            
            System.Diagnostics.Debug.WriteLine($"Drawing tile ({tileData.X}, {tileData.Y}) from RGBA data: {rgbaData.Length} bytes, size {size}x{size}");
            
            // ピクセルサイズを計算（タイルサイズ / データサイズ）
            var pixelWidth = (float)tileData.Size / size;
            var pixelHeight = (float)tileData.Size / size;
            
            // サンプリング間隔（パフォーマンスのため、すべてのピクセルを描画しない）
            var sampleRate = Math.Max(1, size / 64); // 最大64x64で描画
            
            int pixelsDrawn = 0;
            int redPixels = 0, greenPixels = 0, bluePixels = 0;
            
            for (int y = 0; y < size; y += sampleRate)
            {
                for (int x = 0; x < size; x += sampleRate)
                {
                    var pixelIndex = (y * size + x) * 4;
                    if (pixelIndex + 3 < rgbaData.Length)
                    {
                        var r = rgbaData[pixelIndex];
                        var g = rgbaData[pixelIndex + 1];
                        var b = rgbaData[pixelIndex + 2];
                        
                        // デバッグ用カウント
                        if (r > g && r > b && r > 50) redPixels++;
                        else if (g > r && g > b && g > 50) greenPixels++;
                        else if (b > r && b > g && b > 50) bluePixels++;
                        
                        // 黒以外のピクセルのみ描画
                        if (r > 0 || g > 0 || b > 0)
                        {
                            // デバッグ：色値の詳細ログ
                            if (pixelsDrawn < 20) // 最初の20ピクセルのみログ
                            {
                                System.Diagnostics.Debug.WriteLine($"Drawing pixel ({x}, {y}): RGB({r}, {g}, {b})");
                            }
                            
                            var color = Color.FromRgb(r, g, b);
                            
                            // デバッグ：Colorオブジェクトの値を確認
                            if (pixelsDrawn < 5)
                            {
                                System.Diagnostics.Debug.WriteLine($"Color object: R={color.Red}, G={color.Green}, B={color.Blue}, A={color.Alpha}");
                            }
                            
                            // テスト：元の色の代わりに固定色を使用して問題を特定
                            if (pixelsDrawn % 4 == 0)
                                color = Colors.Blue;
                            else if (pixelsDrawn % 4 == 1)
                                color = Colors.Green;
                            else if (pixelsDrawn % 4 == 2)
                                color = Colors.Yellow;
                            // else 元の色を使用
                            
                            canvas.FillColor = color;
                            
                            var pixelX = startX + x * pixelWidth;
                            var pixelY = startY + y * pixelHeight;
                            canvas.FillRectangle(pixelX, pixelY, pixelWidth * sampleRate, pixelHeight * sampleRate);
                            pixelsDrawn++;
                        }
                    }
                }
            }
            
            System.Diagnostics.Debug.WriteLine($"Drew tile ({tileData.X}, {tileData.Y}) using RGBA data: {pixelsDrawn} pixels drawn, Red: {redPixels}, Green: {greenPixels}, Blue: {bluePixels}");
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error drawing tile from RGBA data: {ex.Message}");
        }
    }

    private void DrawDebugInfo(ICanvas canvas, RectF dirtyRect)
    {
        canvas.FontColor = Colors.White;
        canvas.FontSize = 12;
        
        var info = $"Zoom: {_viewport.Parameters.Zoom:F2}x\n" +
                  $"Center: ({_viewport.Parameters.CenterX:F6}, {_viewport.Parameters.CenterY:F6})\n" +
                  $"Iterations: {_viewport.Parameters.MaxIterations}\n" +
                  $"Cache: {_tileManager.CacheCount} tiles";
        
        canvas.DrawString(info, 10, 10, 200, 80, HorizontalAlignment.Left, VerticalAlignment.Top);
    }

    private void OnPanUpdated(object? sender, PanUpdatedEventArgs e)
    {
        switch (e.StatusType)
        {
            case GestureStatus.Started:
                _isDragging = true;
                _lastPanPoint = new Point(e.TotalX, e.TotalY);
                break;
                
            case GestureStatus.Running:
                if (_isDragging)
                {
                    var deltaX = e.TotalX - _lastPanPoint.X;
                    var deltaY = e.TotalY - _lastPanPoint.Y;
                    
                    OnPanGesture(deltaX, deltaY);
                    _lastPanPoint = new Point(e.TotalX, e.TotalY);
                }
                break;
                
            case GestureStatus.Completed:
            case GestureStatus.Canceled:
                _isDragging = false;
                break;
        }
    }

    private void OnPinchUpdated(object? sender, PinchGestureUpdatedEventArgs e)
    {
        if (e.Status == GestureStatus.Running)
        {
            OnZoomGesture(e.Scale, Width / 2, Height / 2);
        }
    }

    private void OnTapped(object? sender, TappedEventArgs e)
    {
        var position = e.GetPosition(this);
        if (position.HasValue)
        {
            // ダブルタップでズームイン
            OnZoomGesture(2.0, position.Value.X, position.Value.Y);
        }
    }

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
        
        var newZoom = _viewport.Parameters.Zoom * zoomFactor;
        _viewport.Parameters.Zoom = Math.Max(0.1, Math.Min(1e15, newZoom));
        
        // ズーム中心を維持
        var (newScreenX, newScreenY) = _viewport.ComplexToScreen(complexX, complexY);
        var deltaX = centerX - newScreenX;
        var deltaY = centerY - newScreenY;
        
        double pixelSize = 1.0 / _viewport.Parameters.Zoom;
        _viewport.Parameters.CenterX -= deltaX * pixelSize;
        _viewport.Parameters.CenterY += deltaY * pixelSize;
        
        ClearOldTiles();
        Invalidate();
    }

    private void ClearOldTiles()
    {
        var currentZoomLevel = GetZoomLevel(_viewport.Parameters.Zoom);
        
        // 古い画像タイルを削除
        var keysToRemove = _renderedTiles.Keys
            .Where(k => Math.Abs(k.ZoomLevel - currentZoomLevel) > 2)
            .ToList();
            
        foreach (var key in keysToRemove)
        {
            if (_renderedTiles.TryRemove(key, out var image))
            {
                image?.Dispose();
            }
        }
        
        // 古いRGBAデータタイルを削除
        var dataKeysToRemove = _renderedTileData.Keys
            .Where(k => Math.Abs(k.ZoomLevel - currentZoomLevel) > 2)
            .ToList();
            
        foreach (var key in dataKeysToRemove)
        {
            _renderedTileData.TryRemove(key, out _);
        }
    }

    private IImage? CreateImageFromRgbaData(byte[] rgbaData, int width, int height)
    {
        try
        {
            System.Diagnostics.Debug.WriteLine($"Creating image from RGBA data: {width}x{height}, data length: {rgbaData.Length}");
            
            // RGBAデータから直接画像を作成するシンプルな方法
            // とりあえずBitmapImageを使用
            var bmpData = CreateBmpFromRgba(rgbaData, width, height);
            System.Diagnostics.Debug.WriteLine($"BMP data created, length: {bmpData.Length}");
            
            var stream = new MemoryStream(bmpData);
            
#if WINDOWS
            // Windowsプラットフォーム用の画像作成
            System.Diagnostics.Debug.WriteLine("Creating image using platform service");
            try
            {
                // より安全な方法で画像を作成
                var platformImage = Microsoft.Maui.Graphics.Platform.PlatformImage.FromStream(stream);
                System.Diagnostics.Debug.WriteLine($"Image created successfully using Platform: {platformImage != null}");
                return platformImage;
            }
            catch (Exception ex2)
            {
                System.Diagnostics.Debug.WriteLine($"Platform image creation failed: {ex2.Message}");
                // フォールバック: 代替方法
                return null;
            }
#else
            return null;
#endif
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Image creation error: {ex.Message}");
            System.Diagnostics.Debug.WriteLine($"Stack trace: {ex.StackTrace}");
            
            try
            {
                var logFile = Path.Combine(AppContext.BaseDirectory, "canvas_debug.log");
                File.AppendAllText(logFile, $"{DateTime.Now:HH:mm:ss.fff} - Image creation error: {ex.Message}\n");
                File.AppendAllText(logFile, $"{DateTime.Now:HH:mm:ss.fff} - Stack trace: {ex.StackTrace}\n");
            }
            catch { /* ログファイル書き込み失敗は無視 */ }
            
            return null;
        }
    }

    private byte[] CreateBmpFromRgba(byte[] rgbaData, int width, int height)
    {
        // BMPヘッダーを作成
        var fileHeaderSize = 14;
        var infoHeaderSize = 40;
        var headerSize = fileHeaderSize + infoHeaderSize;
        var stride = ((width * 3 + 3) / 4) * 4; // 4バイト境界に調整
        var imageSize = stride * height;
        var fileSize = headerSize + imageSize;

        var bmp = new byte[fileSize];
        var offset = 0;

        // BMPファイルヘッダー (14 bytes)
        bmp[offset++] = 0x42; // 'B'
        bmp[offset++] = 0x4D; // 'M'
        BitConverter.GetBytes(fileSize).CopyTo(bmp, offset); offset += 4;
        BitConverter.GetBytes(0).CopyTo(bmp, offset); offset += 4; // Reserved
        BitConverter.GetBytes(headerSize).CopyTo(bmp, offset); offset += 4;

        // BMP情報ヘッダー (40 bytes)
        BitConverter.GetBytes(infoHeaderSize).CopyTo(bmp, offset); offset += 4;
        BitConverter.GetBytes(width).CopyTo(bmp, offset); offset += 4;
        BitConverter.GetBytes(-height).CopyTo(bmp, offset); offset += 4; // 負の値でTop-Down
        BitConverter.GetBytes((short)1).CopyTo(bmp, offset); offset += 2; // Planes
        BitConverter.GetBytes((short)24).CopyTo(bmp, offset); offset += 2; // Bits per pixel
        BitConverter.GetBytes(0).CopyTo(bmp, offset); offset += 4; // Compression
        BitConverter.GetBytes(imageSize).CopyTo(bmp, offset); offset += 4;
        BitConverter.GetBytes(2835).CopyTo(bmp, offset); offset += 4; // X pixels per meter
        BitConverter.GetBytes(2835).CopyTo(bmp, offset); offset += 4; // Y pixels per meter
        BitConverter.GetBytes(0).CopyTo(bmp, offset); offset += 4; // Colors used
        BitConverter.GetBytes(0).CopyTo(bmp, offset); offset += 4; // Important colors

        // ピクセルデータ (BGR形式、各行は4バイト境界)
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var rgbaIndex = (y * width + x) * 4;
                bmp[offset++] = rgbaData[rgbaIndex + 2]; // B
                bmp[offset++] = rgbaData[rgbaIndex + 1]; // G
                bmp[offset++] = rgbaData[rgbaIndex + 0]; // R
                // Alpha チャンネルはスキップ
            }
            
            // 行の残りを0でパディング（4バイト境界に調整）
            while ((offset - headerSize) % 4 != 0)
            {
                bmp[offset++] = 0;
            }
        }

        return bmp;
    }

    private int GetZoomLevel(double zoom) => (int)Math.Floor(Math.Log2(Math.Max(1, zoom)));
}
