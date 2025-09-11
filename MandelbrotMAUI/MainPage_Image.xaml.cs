using MandelbrotMAUI.Services;
using System.Diagnostics;

namespace MandelbrotMAUI;

public partial class MainPage_Image : ContentPage
{
    private readonly IMandelbrotService _mandelbrotService;
    private double _centerX = -0.5;
    private double _centerY = 0.0;
    private double _zoom = 1.0;
    private int _imageWidth = 1024;  // 高解像度
    private int _imageHeight = 1024; // 高解像度
    private bool _isGenerating = false;
    private Point? _lastPanPoint;

    public MainPage_Image()
    {
        InitializeComponent();
        _mandelbrotService = new CudaMandelbrotService();
        
        // ジェスチャー認識を追加
        SetupGestures();
        
        UpdateInfo();
        _ = GenerateImageAsync();
    }

    private void SetupGestures()
    {
        // パンジェスチャー（ドラッグ）
        var panGesture = new PanGestureRecognizer();
        panGesture.PanUpdated += OnPanUpdated;
        MandelbrotImage.GestureRecognizers.Add(panGesture);

        // ピンチジェスチャー（ズーム）
        var pinchGesture = new PinchGestureRecognizer();
        pinchGesture.PinchUpdated += OnPinchUpdated;
        MandelbrotImage.GestureRecognizers.Add(pinchGesture);

        // 左クリック（シングルタップ）でズームイン
        var leftClickGesture = new TapGestureRecognizer { NumberOfTapsRequired = 1 };
        leftClickGesture.Buttons = ButtonsMask.Primary;
        leftClickGesture.Tapped += OnLeftClicked;
        MandelbrotImage.GestureRecognizers.Add(leftClickGesture);

        // 右クリックでズームアウト
        var rightClickGesture = new TapGestureRecognizer();
        rightClickGesture.Buttons = ButtonsMask.Secondary;
        rightClickGesture.Tapped += OnRightClicked;
        MandelbrotImage.GestureRecognizers.Add(rightClickGesture);

        // ダブルクリックで大幅ズームイン
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
            ZoomAtPosition(position.Value, 0.5); // ズームアウト
        }
    }

    private void OnDoubleClicked(object? sender, TappedEventArgs e)
    {
        if (_isGenerating) return;

        var position = e.GetPosition(MandelbrotImage);
        if (position.HasValue)
        {
            ZoomAtPosition(position.Value, 4.0); // 大幅ズームイン
        }
    }

    private void ZoomAtPosition(Point screenPosition, double zoomFactor)
    {
        // 画面座標を複素平面座標に変換
        var complexPosition = ScreenToComplex(screenPosition);
        
        // ズーム実行
        _zoom *= zoomFactor;
        _zoom = Math.Max(0.1, Math.Min(1e15, _zoom));
        
        // 新しいズームレベルでの画面座標を取得
        var newScreenPosition = ComplexToScreen(complexPosition);
        
        // 中心を調整して、クリック位置が変わらないようにする
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
        // 画面サイズを取得
        var imageWidth = MandelbrotImage.Width;
        var imageHeight = MandelbrotImage.Height;
        
        if (imageWidth <= 0 || imageHeight <= 0)
        {
            imageWidth = _imageWidth;
            imageHeight = _imageHeight;
        }

        // 正規化座標（0-1）に変換
        var normalizedX = screenPoint.X / imageWidth;
        var normalizedY = screenPoint.Y / imageHeight;

        // 複素平面の表示範囲を計算
        var aspectRatio = (double)_imageWidth / _imageHeight;
        var range = 4.0 / _zoom;
        
        var rangeX = range * aspectRatio;
        var rangeY = range;

        // 複素平面座標に変換
        var complexX = _centerX + (normalizedX - 0.5) * rangeX;
        var complexY = _centerY - (normalizedY - 0.5) * rangeY; // Y軸反転

        return new Point(complexX, complexY);
    }

    private Point ComplexToScreen(Point complexPoint)
    {
        // 画面サイズを取得
        var imageWidth = MandelbrotImage.Width;
        var imageHeight = MandelbrotImage.Height;
        
        if (imageWidth <= 0 || imageHeight <= 0)
        {
            imageWidth = _imageWidth;
            imageHeight = _imageHeight;
        }

        // 複素平面の表示範囲を計算
        var aspectRatio = (double)_imageWidth / _imageHeight;
        var range = 4.0 / _zoom;
        
        var rangeX = range * aspectRatio;
        var rangeY = range;

        // 正規化座標に変換
        var normalizedX = (complexPoint.X - _centerX) / rangeX + 0.5;
        var normalizedY = -(complexPoint.Y - _centerY) / rangeY + 0.5; // Y軸反転

        // 画面座標に変換
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

                    // 複素平面での移動量を正確に計算
                    var aspectRatio = (double)_imageWidth / _imageHeight;
                    var range = 4.0 / _zoom;
                    
                    var rangeX = range * aspectRatio;
                    var rangeY = range;

                    var imageWidth = MandelbrotImage.Width > 0 ? MandelbrotImage.Width : _imageWidth;
                    var imageHeight = MandelbrotImage.Height > 0 ? MandelbrotImage.Height : _imageHeight;

                    var complexDeltaX = -deltaX * rangeX / imageWidth;
                    var complexDeltaY = deltaY * rangeY / imageHeight; // Y軸反転

                    _centerX += complexDeltaX;
                    _centerY += complexDeltaY;

                    _lastPanPoint = new Point(e.TotalX, e.TotalY);
                    
                    UpdateInfo();
                }
                break;

            case GestureStatus.Completed:
                _lastPanPoint = null;
                _ = GenerateImageAsync(); // パン完了時に再生成
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
            _ = GenerateImageAsync(); // ピンチ完了時に再生成
        }
    }

    private async Task GenerateImageAsync()
    {
        if (_isGenerating) return;
        _isGenerating = true;

        try
        {
            // ズームレベルに応じて反復回数を自動調整
            var adaptiveIterations = CalculateAdaptiveIterations(_zoom);
            
            MainThread.BeginInvokeOnMainThread(() =>
            {
                StatusLabel.Text = $"Computing Mandelbrot set... (Iterations: {adaptiveIterations})";
            });
            
            Debug.WriteLine($"Generating image: {_imageWidth}x{_imageHeight}, Center: ({_centerX}, {_centerY}), Zoom: {_zoom:E2}, Iterations: {adaptiveIterations}");
            
            // CUDA計算を実行
            var rgbaData = await _mandelbrotService.ComputeTileAsync(
                _centerX, _centerY, _zoom, 
                _imageWidth, _imageHeight, adaptiveIterations);
            
            Debug.WriteLine($"Computation completed. Data length: {rgbaData?.Length ?? 0}");
            
            if (rgbaData != null && rgbaData.Length > 0)
            {
                // RGBAデータの色分析
                AnalyzeColors(rgbaData);
                
                // ImageSourceを作成してImageコントロールに設定
                var imageSource = CreateImageSourceFromRgbaData(rgbaData, _imageWidth, _imageHeight);
                
                // UIスレッドで画像を設定
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    MandelbrotImage.Source = imageSource;
                    UpdateInfo(); // 最新の反復回数を表示
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
        // ズームレベルに応じて反復回数を動的に調整
        // 高ズーム時には詳細な境界構造を表示するため反復回数を増加
        var baseIterations = 100;
        var logZoom = Math.Log10(Math.Max(1.0, zoom));
        var adaptiveIterations = (int)(baseIterations + logZoom * 50);
        
        // 最小100、最大2000で制限
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
        
        // 最初の数ピクセルの詳細
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
            // RGBAデータからBMPバイト配列を作成
            var bmpData = CreateBmpFromRgbaData(rgbaData, width, height);
            
            // BMPデータからImageSourceを作成
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

    // 興味深い場所への移動機能
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
