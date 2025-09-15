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
        
        // 繧ｸ繧ｧ繧ｹ繝√Ε繝ｼ隱崎ｭ倥・險ｭ螳・
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
        // 閭梧勹繧帝ｻ偵〒蝪励ｊ縺､縺ｶ縺・
        canvas.FillColor = Colors.Black;
        canvas.FillRectangle(dirtyRect);

        // 繝薙Η繝ｼ繝昴・繝医し繧､繧ｺ繧呈峩譁ｰ
        _viewport.ViewportWidth = dirtyRect.Width;
        _viewport.ViewportHeight = dirtyRect.Height;

        // 陦ｨ遉ｺ遽・峇縺ｮ繧ｿ繧､繝ｫ繧定ｨ育ｮ・
        var visibleTiles = _tileManager.CalculateVisibleTiles(
            dirtyRect.Width, dirtyRect.Height, _viewport);

        // 蜷・ち繧､繝ｫ繧呈緒逕ｻ
        foreach (var tileInfo in visibleTiles)
        {
            DrawTile(canvas, tileInfo);
        }

        // 繝・ヰ繝・げ諠・ｱ繧定｡ｨ遉ｺ
        DrawDebugInfo(canvas, dirtyRect);
    }

    private void DrawTile(ICanvas canvas, TileInfo tileInfo)
    {
        var zoomLevel = GetZoomLevel(_viewport.Parameters.Zoom);
        var tileKey = new TileKey(tileInfo.X, tileInfo.Y, zoomLevel);

        // RGBA繝・・繧ｿ縺ｫ繧医ｋ逶ｴ謗･謠冗判繧偵メ繧ｧ繝・け
        if (_renderedTileData.TryGetValue(tileKey, out var tileData))
        {
            DrawTileFromRgbaData(canvas, tileData);
            return;
        }

        // 繧ｭ繝｣繝・す繝･縺輔ｌ縺溽判蜒上ｒ繝√ぉ繝・け
        if (_renderedTiles.TryGetValue(tileKey, out var image))
        {
            canvas.DrawImage(image, 
                (float)tileInfo.ScreenX, (float)tileInfo.ScreenY, 
                (float)tileInfo.Size, (float)tileInfo.Size);
            return;
        }

        // 繝励Ξ繝ｼ繧ｹ繝帙Ν繝繝ｼ繧呈緒逕ｻ
        canvas.StrokeColor = Colors.Gray;
        canvas.StrokeSize = 1;
        canvas.DrawRectangle((float)tileInfo.ScreenX, (float)tileInfo.ScreenY, 
                           (float)tileInfo.Size, (float)tileInfo.Size);

        // 荳ｭ螟ｮ縺ｫ "Computing..." 繝・く繧ｹ繝医ｒ陦ｨ遉ｺ
        canvas.FontColor = Colors.White;
        canvas.FontSize = 10;
        canvas.DrawString("Computing...", 
            (float)tileInfo.ScreenX + 10, (float)tileInfo.ScreenY + 10, 
            100, 20, HorizontalAlignment.Left, VerticalAlignment.Top);

        // 繧ｿ繧､繝ｫ繝・・繧ｿ繧帝撼蜷梧悄蜿門ｾ・
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

                // CUDA螳溯｣・ RGBA繝・・繧ｿ繧堤峩謗･謠冗判逕ｨ縺ｫ菫晏ｭ・
                if (imageData != null && imageData.Length > 0)
                {
                    System.Diagnostics.Debug.WriteLine($"Creating tile data for ({tileInfo.X}, {tileInfo.Y})");
                    var tileSize = _tileManager.TileSize;
                    
                    // 繝・・繧ｿ繧ｵ繝ｳ繝励Ν繧定ｩｳ縺励￥遒ｺ隱・
                    if (imageData.Length >= 64)
                    {
                        System.Diagnostics.Debug.WriteLine($"RGBA data analysis for tile ({tileInfo.X}, {tileInfo.Y}):");
                        System.Diagnostics.Debug.WriteLine($"Total length: {imageData.Length} bytes");
                        System.Diagnostics.Debug.WriteLine($"Expected size: {tileSize}x{tileSize} = {tileSize * tileSize * 4} bytes");
                        
                        // 隍・焚縺ｮ繧ｵ繝ｳ繝励Ν繝斐け繧ｻ繝ｫ繧堤｢ｺ隱・
                        for (int i = 0; i < Math.Min(16, imageData.Length / 4); i++)
                        {
                            var pixelIndex = i * 4;
                            var r = imageData[pixelIndex];
                            var g = imageData[pixelIndex + 1];
                            var b = imageData[pixelIndex + 2];
                            var a = imageData[pixelIndex + 3];
                            System.Diagnostics.Debug.WriteLine($"Pixel {i}: R={r}, G={g}, B={b}, A={a}");
                        }
                        
                        // 濶ｲ縺ｮ蛻・ｸ・ｒ遒ｺ隱・
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
                    
                    // RGBA繝・・繧ｿ繧偵ち繧､繝ｫ諠・ｱ縺ｨ縺励※菫晏ｭ・
                    var tileRenderData = new TileRenderData
                    {
                        X = tileInfo.X,
                        Y = tileInfo.Y,
                        Size = tileSize,
                        RgbaData = imageData,
                        ScreenX = tileInfo.ScreenX,
                        ScreenY = tileInfo.ScreenY
                    };
                    
                    // 繧ｭ繝｣繝・す繝･縺ｫ霑ｽ蜉・・Image縺ｮ莉｣繧上ｊ縺ｫRGBA繝・・繧ｿ繧剃ｿ晏ｭ假ｼ・
                    _renderedTileData.TryAdd(tileKey, tileRenderData);
                    System.Diagnostics.Debug.WriteLine($"Tile data cached for ({tileInfo.X}, {tileInfo.Y})");
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine($"No image data received for tile ({tileInfo.X}, {tileInfo.Y})");
                }
                
                // UI繧ｹ繝ｬ繝・ラ縺ｧ蜀肴緒逕ｻ
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
                catch { /* 繝ｭ繧ｰ繝輔ぃ繧､繝ｫ譖ｸ縺崎ｾｼ縺ｿ螟ｱ謨励・辟｡隕・*/ }
            }
        });
    }

    private void DrawTileFromRgbaData(ICanvas canvas, TileRenderData tileData)
    {
        try
        {
            // RGBA繝・・繧ｿ繧剃ｽｿ縺｣縺ｦ逶ｴ謗･繝斐け繧ｻ繝ｫ繧呈緒逕ｻ
            var rgbaData = tileData.RgbaData;
            var size = tileData.Size;
            var startX = (float)tileData.ScreenX;
            var startY = (float)tileData.ScreenY;
            
            System.Diagnostics.Debug.WriteLine($"Drawing tile ({tileData.X}, {tileData.Y}) from RGBA data: {rgbaData.Length} bytes, size {size}x{size}");
            
            // 繝斐け繧ｻ繝ｫ繧ｵ繧､繧ｺ繧定ｨ育ｮ暦ｼ医ち繧､繝ｫ繧ｵ繧､繧ｺ / 繝・・繧ｿ繧ｵ繧､繧ｺ・・
            var pixelWidth = (float)tileData.Size / size;
            var pixelHeight = (float)tileData.Size / size;
            
            // 繧ｵ繝ｳ繝励Μ繝ｳ繧ｰ髢馴囈・医ヱ繝輔か繝ｼ繝槭Φ繧ｹ縺ｮ縺溘ａ縲√☆縺ｹ縺ｦ縺ｮ繝斐け繧ｻ繝ｫ繧呈緒逕ｻ縺励↑縺・ｼ・
            var sampleRate = Math.Max(1, size / 64); // 譛螟ｧ64x64縺ｧ謠冗判
            
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
                        
                        // 繝・ヰ繝・げ逕ｨ繧ｫ繧ｦ繝ｳ繝・
                        if (r > g && r > b && r > 50) redPixels++;
                        else if (g > r && g > b && g > 50) greenPixels++;
                        else if (b > r && b > g && b > 50) bluePixels++;
                        
                        // 鮟剃ｻ･螟悶・繝斐け繧ｻ繝ｫ縺ｮ縺ｿ謠冗判
                        if (r > 0 || g > 0 || b > 0)
                        {
                            // 繝・ヰ繝・げ・夊牡蛟､縺ｮ隧ｳ邏ｰ繝ｭ繧ｰ
                            if (pixelsDrawn < 20) // 譛蛻昴・20繝斐け繧ｻ繝ｫ縺ｮ縺ｿ繝ｭ繧ｰ
                            {
                                System.Diagnostics.Debug.WriteLine($"Drawing pixel ({x}, {y}): RGB({r}, {g}, {b})");
                            }
                            
                            var color = Color.FromRgb(r, g, b);
                            
                            // 繝・ヰ繝・げ・咾olor繧ｪ繝悶ず繧ｧ繧ｯ繝医・蛟､繧堤｢ｺ隱・
                            if (pixelsDrawn < 5)
                            {
                                System.Diagnostics.Debug.WriteLine($"Color object: R={color.Red}, G={color.Green}, B={color.Blue}, A={color.Alpha}");
                            }
                            
                            // 繝・せ繝茨ｼ壼・縺ｮ濶ｲ縺ｮ莉｣繧上ｊ縺ｫ蝗ｺ螳夊牡繧剃ｽｿ逕ｨ縺励※蝠城｡後ｒ迚ｹ螳・
                            if (pixelsDrawn % 4 == 0)
                                color = Colors.Blue;
                            else if (pixelsDrawn % 4 == 1)
                                color = Colors.Green;
                            else if (pixelsDrawn % 4 == 2)
                                color = Colors.Yellow;
                            // else 蜈・・濶ｲ繧剃ｽｿ逕ｨ
                            
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
            // 繝繝悶Ν繧ｿ繝・・縺ｧ繧ｺ繝ｼ繝繧､繝ｳ
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
        
        // 繧ｺ繝ｼ繝荳ｭ蠢・ｒ邯ｭ謖・
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
        
        // 蜿､縺・判蜒上ち繧､繝ｫ繧貞炎髯､
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
        
        // 蜿､縺СGBA繝・・繧ｿ繧ｿ繧､繝ｫ繧貞炎髯､
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
            
            // RGBA繝・・繧ｿ縺九ｉ逶ｴ謗･逕ｻ蜒上ｒ菴懈・縺吶ｋ繧ｷ繝ｳ繝励Ν縺ｪ譁ｹ豕・
            // 縺ｨ繧翫≠縺医★BitmapImage繧剃ｽｿ逕ｨ
            var bmpData = CreateBmpFromRgba(rgbaData, width, height);
            System.Diagnostics.Debug.WriteLine($"BMP data created, length: {bmpData.Length}");
            
            var stream = new MemoryStream(bmpData);
            
#if WINDOWS
            // Windows繝励Λ繝・ヨ繝輔か繝ｼ繝逕ｨ縺ｮ逕ｻ蜒丈ｽ懈・
            System.Diagnostics.Debug.WriteLine("Creating image using platform service");
            try
            {
                // 繧医ｊ螳牙・縺ｪ譁ｹ豕輔〒逕ｻ蜒上ｒ菴懈・
                var platformImage = Microsoft.Maui.Graphics.Platform.PlatformImage.FromStream(stream);
                System.Diagnostics.Debug.WriteLine($"Image created successfully using Platform: {platformImage != null}");
                return platformImage;
            }
            catch (Exception ex2)
            {
                System.Diagnostics.Debug.WriteLine($"Platform image creation failed: {ex2.Message}");
                // 繝輔か繝ｼ繝ｫ繝舌ャ繧ｯ: 莉｣譖ｿ譁ｹ豕・
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
            catch { /* 繝ｭ繧ｰ繝輔ぃ繧､繝ｫ譖ｸ縺崎ｾｼ縺ｿ螟ｱ謨励・辟｡隕・*/ }
            
            return null;
        }
    }

    private byte[] CreateBmpFromRgba(byte[] rgbaData, int width, int height)
    {
        // BMP繝倥ャ繝繝ｼ繧剃ｽ懈・
        var fileHeaderSize = 14;
        var infoHeaderSize = 40;
        var headerSize = fileHeaderSize + infoHeaderSize;
        var stride = ((width * 3 + 3) / 4) * 4; // 4繝舌う繝亥｢・阜縺ｫ隱ｿ謨ｴ
        var imageSize = stride * height;
        var fileSize = headerSize + imageSize;

        var bmp = new byte[fileSize];
        var offset = 0;

        // BMP繝輔ぃ繧､繝ｫ繝倥ャ繝繝ｼ (14 bytes)
        bmp[offset++] = 0x42; // 'B'
        bmp[offset++] = 0x4D; // 'M'
        BitConverter.GetBytes(fileSize).CopyTo(bmp, offset); offset += 4;
        BitConverter.GetBytes(0).CopyTo(bmp, offset); offset += 4; // Reserved
        BitConverter.GetBytes(headerSize).CopyTo(bmp, offset); offset += 4;

        // BMP諠・ｱ繝倥ャ繝繝ｼ (40 bytes)
        BitConverter.GetBytes(infoHeaderSize).CopyTo(bmp, offset); offset += 4;
        BitConverter.GetBytes(width).CopyTo(bmp, offset); offset += 4;
        BitConverter.GetBytes(-height).CopyTo(bmp, offset); offset += 4; // 雋縺ｮ蛟､縺ｧTop-Down
        BitConverter.GetBytes((short)1).CopyTo(bmp, offset); offset += 2; // Planes
        BitConverter.GetBytes((short)24).CopyTo(bmp, offset); offset += 2; // Bits per pixel
        BitConverter.GetBytes(0).CopyTo(bmp, offset); offset += 4; // Compression
        BitConverter.GetBytes(imageSize).CopyTo(bmp, offset); offset += 4;
        BitConverter.GetBytes(2835).CopyTo(bmp, offset); offset += 4; // X pixels per meter
        BitConverter.GetBytes(2835).CopyTo(bmp, offset); offset += 4; // Y pixels per meter
        BitConverter.GetBytes(0).CopyTo(bmp, offset); offset += 4; // Colors used
        BitConverter.GetBytes(0).CopyTo(bmp, offset); offset += 4; // Important colors

        // 繝斐け繧ｻ繝ｫ繝・・繧ｿ (BGR蠖｢蠑上∝推陦後・4繝舌う繝亥｢・阜)
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var rgbaIndex = (y * width + x) * 4;
                bmp[offset++] = rgbaData[rgbaIndex + 2]; // B
                bmp[offset++] = rgbaData[rgbaIndex + 1]; // G
                bmp[offset++] = rgbaData[rgbaIndex + 0]; // R
                // Alpha 繝√Ε繝ｳ繝阪Ν縺ｯ繧ｹ繧ｭ繝・・
            }
            
            // 陦後・谿九ｊ繧・縺ｧ繝代ョ繧｣繝ｳ繧ｰ・・繝舌う繝亥｢・阜縺ｫ隱ｿ謨ｴ・・
            while ((offset - headerSize) % 4 != 0)
            {
                bmp[offset++] = 0;
            }
        }

        return bmp;
    }

    private int GetZoomLevel(double zoom) => (int)Math.Floor(Math.Log2(Math.Max(1, zoom)));
}
