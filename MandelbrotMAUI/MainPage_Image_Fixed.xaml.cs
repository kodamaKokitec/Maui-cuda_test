using MandelbrotMAUI.Services;
using System.Diagnostics;

namespace MandelbrotMAUI;

public partial class MainPage_Image : ContentPage
{
    private readonly IMandelbrotService _mandelbrotService;
    private double _centerX = -0.5;
    private double _centerY = 0.0;
    private double _zoom = 1.0;
    private int _imageWidth = 1024;  // 鬮倩ｧ｣蜒丞ｺｦ
    private int _imageHeight = 1024; // 鬮倩ｧ｣蜒丞ｺｦ
    private bool _isGenerating = false;
    private Point? _lastPanPoint;

    public MainPage_Image()
    {
        InitializeComponent();
        _mandelbrotService = new CudaMandelbrotService();
        
        // 繧ｸ繧ｧ繧ｹ繝√Ε繝ｼ隱崎ｭ倥ｒ霑ｽ蜉
        SetupGestures();
        
        UpdateInfo();
        _ = GenerateImageAsync();
    }

    private void SetupGestures()
    {
        // 繝代Φ繧ｸ繧ｧ繧ｹ繝√Ε繝ｼ・医ラ繝ｩ繝・げ・・
        var panGesture = new PanGestureRecognizer();
        panGesture.PanUpdated += OnPanUpdated;
        MandelbrotImage.GestureRecognizers.Add(panGesture);

        // 繝斐Φ繝√ず繧ｧ繧ｹ繝√Ε繝ｼ・医ぜ繝ｼ繝・・
        var pinchGesture = new PinchGestureRecognizer();
        pinchGesture.PinchUpdated += OnPinchUpdated;
        MandelbrotImage.GestureRecognizers.Add(pinchGesture);

        // 蟾ｦ繧ｯ繝ｪ繝・け・医す繝ｳ繧ｰ繝ｫ繧ｿ繝・・・峨〒繧ｺ繝ｼ繝繧､繝ｳ
        var leftClickGesture = new TapGestureRecognizer { NumberOfTapsRequired = 1 };
        leftClickGesture.Buttons = ButtonsMask.Primary;
        leftClickGesture.Tapped += OnLeftClicked;
        MandelbrotImage.GestureRecognizers.Add(leftClickGesture);

        // 蜿ｳ繧ｯ繝ｪ繝・け縺ｧ繧ｺ繝ｼ繝繧｢繧ｦ繝・
        var rightClickGesture = new TapGestureRecognizer();
        rightClickGesture.Buttons = ButtonsMask.Secondary;
        rightClickGesture.Tapped += OnRightClicked;
        MandelbrotImage.GestureRecognizers.Add(rightClickGesture);

        // 繝繝悶Ν繧ｯ繝ｪ繝・け縺ｧ螟ｧ蟷・ぜ繝ｼ繝繧､繝ｳ
        var doubleClickGesture = new TapGestureRecognizer { NumberOfTapsRequired = 2 };
        doubleClickGesture.Tapped += OnDoubleClicked;
        MandelbrotImage.GestureRecognizers.Add(doubleClickGesture);
    }

    private void OnLeftClicked(object? sender, TappedEventArgs e)
    {
        if (_isGenerating) return;

        var position = e.GetPosition(MandelbrotImage);
        if (position.HasValue)
        {
            ZoomAtPosition(position.Value, 2.0);
        }
    }

    private void OnRightClicked(object? sender, TappedEventArgs e)
    {
        if (_isGenerating) return;

        var position = e.GetPosition(MandelbrotImage);
        if (position.HasValue)
        {
            ZoomAtPosition(position.Value, 0.5); // 繧ｺ繝ｼ繝繧｢繧ｦ繝・
        }
    }

    private void OnDoubleClicked(object? sender, TappedEventArgs e)
    {
        if (_isGenerating) return;

        var position = e.GetPosition(MandelbrotImage);
        if (position.HasValue)
        {
            ZoomAtPosition(position.Value, 4.0); // 螟ｧ蟷・ぜ繝ｼ繝繧､繝ｳ
        }
    }

    private void ZoomAtPosition(Point screenPosition, double zoomFactor)
    {
        // 逕ｻ髱｢蠎ｧ讓吶ｒ隍・ｴ蟷ｳ髱｢蠎ｧ讓吶↓螟画鋤
        var complexPosition = ScreenToComplex(screenPosition);
        
        // 繧ｺ繝ｼ繝螳溯｡・
        _zoom *= zoomFactor;
        _zoom = Math.Max(0.1, Math.Min(1e15, _zoom));
        
        // 譁ｰ縺励＞繧ｺ繝ｼ繝繝ｬ繝吶Ν縺ｧ縺ｮ逕ｻ髱｢蠎ｧ讓吶ｒ蜿門ｾ・
        var newScreenPosition = ComplexToScreen(complexPosition);
        
        // 荳ｭ蠢・ｒ隱ｿ謨ｴ縺励※縲√け繝ｪ繝・け菴咲ｽｮ縺悟､峨ｏ繧峨↑縺・ｈ縺・↓縺吶ｋ
        var screenCenter = new Point(MandelbrotImage.Width / 2, MandelbrotImage.Height / 2);
        var offset = ScreenToComplex(new Point(
            screenCenter.X + (screenPosition.X - newScreenPosition.X),
            screenCenter.Y + (screenPosition.Y - newScreenPosition.Y)
        ));
        
        _centerX = offset.X;
        _centerY = offset.Y;

        UpdateInfo();
        _ = GenerateImageAsync();
    }

    private Point ScreenToComplex(Point screenPoint)
    {
        // 逕ｻ髱｢繧ｵ繧､繧ｺ繧貞叙蠕・
        var imageWidth = MandelbrotImage.Width;
        var imageHeight = MandelbrotImage.Height;
        
        if (imageWidth <= 0 || imageHeight <= 0)
        {
            imageWidth = _imageWidth;
            imageHeight = _imageHeight;
        }

        // 豁｣隕丞喧蠎ｧ讓呻ｼ・-1・峨↓螟画鋤
        var normalizedX = screenPoint.X / imageWidth;
        var normalizedY = screenPoint.Y / imageHeight;

        // 隍・ｴ蟷ｳ髱｢縺ｮ陦ｨ遉ｺ遽・峇繧定ｨ育ｮ・
        var aspectRatio = (double)_imageWidth / _imageHeight;
        var range = 4.0 / _zoom;
        
        var rangeX = range * aspectRatio;
        var rangeY = range;

        // 隍・ｴ蟷ｳ髱｢蠎ｧ讓吶↓螟画鋤
        var complexX = _centerX + (normalizedX - 0.5) * rangeX;
        var complexY = _centerY - (normalizedY - 0.5) * rangeY; // Y霆ｸ蜿崎ｻ｢

        return new Point(complexX, complexY);
    }

    private Point ComplexToScreen(Point complexPoint)
    {
        // 逕ｻ髱｢繧ｵ繧､繧ｺ繧貞叙蠕・
        var imageWidth = MandelbrotImage.Width;
        var imageHeight = MandelbrotImage.Height;
        
        if (imageWidth <= 0 || imageHeight <= 0)
        {
            imageWidth = _imageWidth;
            imageHeight = _imageHeight;
        }

        // 隍・ｴ蟷ｳ髱｢縺ｮ陦ｨ遉ｺ遽・峇繧定ｨ育ｮ・
        var aspectRatio = (double)_imageWidth / _imageHeight;
        var range = 4.0 / _zoom;
        
        var rangeX = range * aspectRatio;
        var rangeY = range;

        // 豁｣隕丞喧蠎ｧ讓吶↓螟画鋤
        var normalizedX = (complexPoint.X - _centerX) / rangeX + 0.5;
        var normalizedY = -(complexPoint.Y - _centerY) / rangeY + 0.5; // Y霆ｸ蜿崎ｻ｢

        // 逕ｻ髱｢蠎ｧ讓吶↓螟画鋤
        var screenX = normalizedX * imageWidth;
        var screenY = normalizedY * imageHeight;

        return new Point(screenX, screenY);
    }

    private void UpdateInfo()
    {
        var adaptiveIterations = CalculateAdaptiveIterations(_zoom);
        InfoLabel.Text = $"Zoom: {_zoom:E2}x, Center: ({_centerX:F6}, {_centerY:F6}), Iterations: {adaptiveIterations}";
        StatusLabel.Text = $"Engine: {_mandelbrotService.GetEngineInfo()}";
    }

    private void OnPanUpdated(object? sender, PanUpdatedEventArgs e)
    {
        if (_isGenerating) return;

        switch (e.StatusType)
        {
            case GestureStatus.Started:
                _lastPanPoint = new Point(e.TotalX, e.TotalY);
                break;

            case GestureStatus.Running:
                if (_lastPanPoint.HasValue)
                {
                    var deltaX = e.TotalX - _lastPanPoint.Value.X;
                    var deltaY = e.TotalY - _lastPanPoint.Value.Y;

                    // 隍・ｴ蟷ｳ髱｢縺ｧ縺ｮ遘ｻ蜍暮㍼繧呈ｭ｣遒ｺ縺ｫ險育ｮ・
                    var aspectRatio = (double)_imageWidth / _imageHeight;
                    var range = 4.0 / _zoom;
                    
                    var rangeX = range * aspectRatio;
                    var rangeY = range;

                    var imageWidth = MandelbrotImage.Width > 0 ? MandelbrotImage.Width : _imageWidth;
                    var imageHeight = MandelbrotImage.Height > 0 ? MandelbrotImage.Height : _imageHeight;

                    var complexDeltaX = -deltaX * rangeX / imageWidth;
                    var complexDeltaY = deltaY * rangeY / imageHeight; // Y霆ｸ蜿崎ｻ｢

                    _centerX += complexDeltaX;
                    _centerY += complexDeltaY;

                    _lastPanPoint = new Point(e.TotalX, e.TotalY);
                    
                    UpdateInfo();
                }
                break;

            case GestureStatus.Completed:
                _lastPanPoint = null;
                _ = GenerateImageAsync(); // 繝代Φ螳御ｺ・凾縺ｫ蜀咲函謌・
                break;
        }
    }

    private void OnPinchUpdated(object? sender, PinchGestureUpdatedEventArgs e)
    {
        if (_isGenerating) return;

        if (e.Status == GestureStatus.Running)
        {
            var newZoom = _zoom * e.Scale;
            _zoom = Math.Max(0.1, Math.Min(1e15, newZoom));
            UpdateInfo();
        }
        else if (e.Status == GestureStatus.Completed)
        {
            _ = GenerateImageAsync(); // 繝斐Φ繝∝ｮ御ｺ・凾縺ｫ蜀咲函謌・
        }
    }

    private async Task GenerateImageAsync()
    {
        if (_isGenerating) return;
        _isGenerating = true;

        try
        {
            // 繧ｺ繝ｼ繝繝ｬ繝吶Ν縺ｫ蠢懊§縺ｦ蜿榊ｾｩ蝗樊焚繧定・蜍戊ｪｿ謨ｴ
            var adaptiveIterations = CalculateAdaptiveIterations(_zoom);
            
            MainThread.BeginInvokeOnMainThread(() =>
            {
                StatusLabel.Text = $"Computing Mandelbrot set... (Iterations: {adaptiveIterations})";
            });
            
            Debug.WriteLine($"Generating image: {_imageWidth}x{_imageHeight}, Center: ({_centerX}, {_centerY}), Zoom: {_zoom:E2}, Iterations: {adaptiveIterations}");
            
            // CUDA險育ｮ励ｒ螳溯｡・
            var rgbaData = await _mandelbrotService.ComputeTileAsync(
                _centerX, _centerY, _zoom, 
                _imageWidth, _imageHeight, adaptiveIterations);
            
            Debug.WriteLine($"Computation completed. Data length: {rgbaData?.Length ?? 0}");
            
            if (rgbaData != null && rgbaData.Length > 0)
            {
                // RGBA繝・・繧ｿ縺ｮ濶ｲ蛻・梵
                AnalyzeColors(rgbaData);
                
                // ImageSource繧剃ｽ懈・縺励※Image繧ｳ繝ｳ繝医Ο繝ｼ繝ｫ縺ｫ險ｭ螳・
                var imageSource = CreateImageSourceFromRgbaData(rgbaData, _imageWidth, _imageHeight);
                
                // UI繧ｹ繝ｬ繝・ラ縺ｧ逕ｻ蜒上ｒ險ｭ螳・
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    MandelbrotImage.Source = imageSource;
                    UpdateInfo(); // 譛譁ｰ縺ｮ蜿榊ｾｩ蝗樊焚繧定｡ｨ遉ｺ
                    StatusLabel.Text = $"Generated successfully! ({adaptiveIterations} iterations)";
                });
            }
            else
            {
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    StatusLabel.Text = "Failed to generate image data";
                });
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Error generating image: {ex.Message}");
            MainThread.BeginInvokeOnMainThread(() =>
            {
                StatusLabel.Text = $"Error: {ex.Message}";
            });
        }
        finally
        {
            _isGenerating = false;
        }
    }

    private int CalculateAdaptiveIterations(double zoom)
    {
        // 繧ｺ繝ｼ繝繝ｬ繝吶Ν縺ｫ蠢懊§縺ｦ蜿榊ｾｩ蝗樊焚繧貞虚逧・↓隱ｿ謨ｴ
        // 鬮倥ぜ繝ｼ繝譎ゅ↓縺ｯ隧ｳ邏ｰ縺ｪ蠅・阜讒矩繧定｡ｨ遉ｺ縺吶ｋ縺溘ａ蜿榊ｾｩ蝗樊焚繧貞｢怜刈
        var baseIterations = 100;
        var logZoom = Math.Log10(Math.Max(1.0, zoom));
        var adaptiveIterations = (int)(baseIterations + logZoom * 50);
        
        // 譛蟆・00縲∵怙螟ｧ2000縺ｧ蛻ｶ髯・
        return Math.Max(100, Math.Min(2000, adaptiveIterations));
    }

    private void AnalyzeColors(byte[] rgbaData)
    {
        int redCount = 0, greenCount = 0, blueCount = 0, blackCount = 0, otherCount = 0;
        
        for (int i = 0; i < rgbaData.Length; i += 4)
        {
            var r = rgbaData[i];
            var g = rgbaData[i + 1];
            var b = rgbaData[i + 2];
            
            if (r == 0 && g == 0 && b == 0)
                blackCount++;
            else if (r > g && r > b && r > 50)
                redCount++;
            else if (g > r && g > b && g > 50)
                greenCount++;
            else if (b > r && b > g && b > 50)
                blueCount++;
            else
                otherCount++;
        }
        
        var totalPixels = rgbaData.Length / 4;
        Debug.WriteLine($"Color analysis - Total: {totalPixels}, Red: {redCount}, Green: {greenCount}, Blue: {blueCount}, Black: {blackCount}, Other: {otherCount}");
        
        // 譛蛻昴・謨ｰ繝斐け繧ｻ繝ｫ縺ｮ隧ｳ邏ｰ
        Debug.WriteLine("First 10 pixels:");
        for (int i = 0; i < Math.Min(10, rgbaData.Length / 4); i++)
        {
            var idx = i * 4;
            Debug.WriteLine($"Pixel {i}: RGBA({rgbaData[idx]}, {rgbaData[idx + 1]}, {rgbaData[idx + 2]}, {rgbaData[idx + 3]})");
        }
    }

    private ImageSource CreateImageSourceFromRgbaData(byte[] rgbaData, int width, int height)
    {
        try
        {
            // RGBA繝・・繧ｿ縺九ｉBMP繝舌う繝磯・蛻励ｒ菴懈・
            var bmpData = CreateBmpFromRgbaData(rgbaData, width, height);
            
            // BMP繝・・繧ｿ縺九ｉImageSource繧剃ｽ懈・
            return ImageSource.FromStream(() => new MemoryStream(bmpData));
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Error creating ImageSource: {ex.Message}");
            return null;
        }
    }

    private byte[] CreateBmpFromRgbaData(byte[] rgbaData, int width, int height)
    {
        // BMP file format
        int imageSize = width * height * 3; // RGB (24-bit)
        int fileSize = 54 + imageSize; // BMP header is 54 bytes
        
        using (var ms = new MemoryStream())
        using (var writer = new BinaryWriter(ms))
        {
            // BMP file header (14 bytes)
            writer.Write((byte)'B');
            writer.Write((byte)'M');
            writer.Write(fileSize);        // File size
            writer.Write((int)0);          // Reserved
            writer.Write(54);              // Offset to image data
            
            // BMP info header (40 bytes)
            writer.Write(40);              // Info header size
            writer.Write(width);           // Image width
            writer.Write(height);          // Image height
            writer.Write((short)1);        // Planes
            writer.Write((short)24);       // Bits per pixel
            writer.Write(0);               // Compression
            writer.Write(imageSize);       // Image size
            writer.Write(0);               // X pixels per meter
            writer.Write(0);               // Y pixels per meter
            writer.Write(0);               // Colors used
            writer.Write(0);               // Important colors
            
            // BMP data is stored bottom-to-top, so we need to flip the image
            for (int y = height - 1; y >= 0; y--)
            {
                for (int x = 0; x < width; x++)
                {
                    int srcIndex = (y * width + x) * 4; // RGBA
                    // BMP uses BGR order, while our data is RGBA
                    writer.Write(rgbaData[srcIndex + 2]); // B
                    writer.Write(rgbaData[srcIndex + 1]); // G
                    writer.Write(rgbaData[srcIndex]);     // R (skip A)
                }
            }
            
            return ms.ToArray();
        }
    }

    // 闊亥袖豺ｱ縺・ｴ謇縺ｸ縺ｮ遘ｻ蜍墓ｩ溯・
    public async Task JumpToInterestingLocation(int locationIndex)
    {
        if (_isGenerating) return;

        var locations = new[]
        {
            new { Name = "Main Set", X = -0.5, Y = 0.0, Zoom = 1.0 },
            new { Name = "Seahorse Valley", X = -0.75, Y = 0.1, Zoom = 100.0 },
            new { Name = "Lightning", X = -1.775, Y = 0.0, Zoom = 1000.0 },
            new { Name = "Spiral", X = -0.7269, Y = 0.1889, Zoom = 10000.0 },
            new { Name = "Mini Mandelbrot", X = -0.16, Y = 1.0405, Zoom = 100000.0 },
            new { Name = "Dragon", X = -0.8, Y = 0.156, Zoom = 50000.0 }
        };

        if (locationIndex >= 0 && locationIndex < locations.Length)
        {
            var location = locations[locationIndex];
            _centerX = location.X;
            _centerY = location.Y;
            _zoom = location.Zoom;

            UpdateInfo();
            await GenerateImageAsync();
        }
    }
}
