using MandelbrotMAUI.Models;
using MandelbrotMAUI.Services;
using System.Diagnostics;
using System.Numerics;

namespace MandelbrotMAUI;

public partial class MainPage_Image : ContentPage
{
    // マンデルブロー集合の興味深い特徴点
    public class InterestingPoint
    {
        public string Name { get; set; } = "";
        public double X { get; set; }
        public double Y { get; set; }
        public double OptimalZoom { get; set; }
        public string Description { get; set; } = "";
    }

    private static readonly List<InterestingPoint> InterestingPoints = new()
    {
        new InterestingPoint { Name = "メインカルディオイド", X = -0.5, Y = 0.0, OptimalZoom = 5.0, Description = "マンデルブロー集合の中心部" },
        new InterestingPoint { Name = "左の大きなバルブ", X = -1.0, Y = 0.0, OptimalZoom = 10.0, Description = "左側の大きな円形領域" },
        new InterestingPoint { Name = "上部のスパイラル", X = -0.16, Y = 1.03, OptimalZoom = 500.0, Description = "美しいスパイラル構造" },
        new InterestingPoint { Name = "下部のスパイラル", X = -0.16, Y = -1.03, OptimalZoom = 500.0, Description = "下側のスパイラル構造" },
        new InterestingPoint { Name = "ミニマンデルブロー1", X = -0.7269, Y = 0.1889, OptimalZoom = 2000.0, Description = "小さなマンデルブロー集合のコピー" },
        new InterestingPoint { Name = "ミニマンデルブロー2", X = -0.8, Y = 0.156, OptimalZoom = 5000.0, Description = "非常に小さなマンデルブロー集合" },
        new InterestingPoint { Name = "シーホース バレー", X = -0.7463, Y = 0.1102, OptimalZoom = 10000.0, Description = "タツノオトシゴのような形状" },
        new InterestingPoint { Name = "エレファント バレー", X = 0.25, Y = 0.0, OptimalZoom = 100.0, Description = "象のような形の谷" },
        new InterestingPoint { Name = "ライトニング", X = -1.775, Y = 0.0, OptimalZoom = 150.0, Description = "稲妻のような形状" }
    };

    private readonly IMandelbrotService _mandelbrotService;
    private double _centerX = -0.5;
    private double _centerY = 0.0;
    private double _zoom = 1.0;
    private int _imageWidth = 4096;  // 4K解像度で非常に詳細な表示
    private int _imageHeight = 4096; // 4K解像度で非常に詳細な表示
    private bool _isGenerating = false;
    private Point? _lastPanPoint;
    
    // 画像の実際の表示サイズを追跡
    private double _actualImageWidth = 800;
    private double _actualImageHeight = 800;

    public MainPage_Image()
    {
        InitializeComponent();
        _mandelbrotService = new CudaMandelbrotService();
        
        // 画像のサイズ変更イベントを監視
        MandelbrotImage.SizeChanged += OnImageSizeChanged;
        
        // ジェスチャー認識を追加
        SetupGestures();
        
        // 初期表示を全体の特徴点が見えるように調整
        SetOptimalInitialView();
        
        UpdateInfo();
        _ = GenerateImageAsync();
    }

    // 全体の特徴点が見える最適な初期表示を設定
    private void SetOptimalInitialView()
    {
        // 全ての特徴点の境界を計算
        var minX = InterestingPoints.Min(p => p.X) - 0.5; // マージンを追加
        var maxX = InterestingPoints.Max(p => p.X) + 0.5;
        var minY = InterestingPoints.Min(p => p.Y) - 0.5;
        var maxY = InterestingPoints.Max(p => p.Y) + 0.5;
        
        // 中心を計算
        _centerX = (minX + maxX) / 2.0;
        _centerY = (minY + maxY) / 2.0;
        
        // 全体が見えるズームレベルを計算
        var rangeX = maxX - minX;
        var rangeY = maxY - minY;
        var maxRange = Math.Max(rangeX, rangeY);
        
        // 4.0は標準的な表示範囲、少し余裕を持たせる
        _zoom = 4.0 / (maxRange * 1.2);
    }

    // 画像の実際の表示サイズが変更された時のイベントハンドラー
    private void OnImageSizeChanged(object? sender, EventArgs e)
    {
        if (sender is Image image)
        {
            _actualImageWidth = image.Width;
            _actualImageHeight = image.Height;
            
            // 座標情報を更新
            UpdateInfo();
        }
    }

    private void SetupGestures()
    {
        // パンジェスチャー�E�ドラチE���E�E
        var panGesture = new PanGestureRecognizer();
        panGesture.PanUpdated += OnPanUpdated;
        MandelbrotImage.GestureRecognizers.Add(panGesture);

        // ピンチジェスチャー�E�ズーム�E�E
        var pinchGesture = new PinchGestureRecognizer();
        pinchGesture.PinchUpdated += OnPinchUpdated;
        MandelbrotImage.GestureRecognizers.Add(pinchGesture);

        // 左クリチE���E�シングルタチE�E�E�でズームイン
        var leftClickGesture = new TapGestureRecognizer { NumberOfTapsRequired = 1 };
        leftClickGesture.Buttons = ButtonsMask.Primary;
        leftClickGesture.Tapped += OnLeftClicked;
        MandelbrotImage.GestureRecognizers.Add(leftClickGesture);

        // 右クリチE��でズームアウチE
        var rightClickGesture = new TapGestureRecognizer();
        rightClickGesture.Buttons = ButtonsMask.Secondary;
        rightClickGesture.Tapped += OnRightClicked;
        MandelbrotImage.GestureRecognizers.Add(rightClickGesture);

        // ダブルクリチE��で大幁E��ームイン
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
            ZoomAtPosition(position.Value, 0.5); // ズームアウチE
        }
    }

    private void OnDoubleClicked(object? sender, TappedEventArgs e)
    {
        if (_isGenerating) return;

        var position = e.GetPosition(MandelbrotImage);
        if (position.HasValue)
        {
            ZoomAtPosition(position.Value, 4.0); // 大幁E��ームイン
        }
    }

    // 全体表示ボタンのイベントハンドラー
    private void OnResetViewClicked(object? sender, EventArgs e)
    {
        if (_isGenerating) return;
        
        SetOptimalInitialView();
        UpdateInfo();
        _ = GenerateImageAsync();
    }

    // 次の特徴点ボタンのイベントハンドラー
    private static int _currentFeatureIndex = 0;
    private void OnNextFeatureClicked(object? sender, EventArgs e)
    {
        if (_isGenerating) return;
        
        _currentFeatureIndex = (_currentFeatureIndex + 1) % InterestingPoints.Count;
        var feature = InterestingPoints[_currentFeatureIndex];
        
        _centerX = feature.X;
        _centerY = feature.Y;
        _zoom = feature.OptimalZoom;
        
        UpdateInfo();
        _ = GenerateImageAsync();
    }

    private void ZoomAtPosition(Point screenPosition, double zoomFactor)
    {
        // クリック点の複素平面座標を取得（ズーム前）
        var clickComplex = ScreenToComplex(screenPosition);
        
        // 新しいズームレベルを計算
        var newZoom = _zoom * zoomFactor;
        var maxAllowedZoom = CalculateMaxZoomForInterestingPoints();
        newZoom = Math.Max(0.1, Math.Min(maxAllowedZoom, newZoom));
        
        // クリック点を中心にズームするため、新しい中心を計算
        // クリック点が画面中央に来るように中心を移動
        _centerX = clickComplex.Real;
        _centerY = clickComplex.Imaginary;
        _zoom = newZoom;
        
        UpdateInfo();
        _ = GenerateImageAsync();
    }

    // 特徴点が画面から外れない最大ズームレベルを計算（大幅に緩和）
    private double CalculateMaxZoomForInterestingPoints()
    {
        // 現在の表示領域内にある特徴点を検索
        var range = 4.0 / _zoom;
        var halfRange = range / 2.0;
        
        var nearbyPoints = InterestingPoints.Where(p => 
            Math.Abs(p.X - _centerX) <= halfRange * 2.0 && 
            Math.Abs(p.Y - _centerY) <= halfRange * 2.0).ToList();
        
        if (nearbyPoints.Any())
        {
            // 近くに特徴点がある場合は、その特徴点の最適ズームレベルの10倍まで許可
            var nearestPoint = nearbyPoints
                .OrderBy(p => Math.Sqrt(Math.Pow(p.X - _centerX, 2) + Math.Pow(p.Y - _centerY, 2)))
                .First();
            return Math.Min(1e12, nearestPoint.OptimalZoom * 10.0);
        }
        
        // 遠い場合は従来の制限を適用（ただし大幅に緩和）
        var maxDistance = 0.0;
        foreach (var point in InterestingPoints)
        {
            var distance = Math.Sqrt(Math.Pow(point.X - _centerX, 2) + Math.Pow(point.Y - _centerY, 2));
            maxDistance = Math.Max(maxDistance, distance);
        }
        
        // より深いズームを許可（従来の10倍）
        var maxZoom = 20.0 / maxDistance;
        return Math.Min(1e12, maxZoom);
    }

    // 指定された中心とズームで特徴点が表示範囲内にあるかチェック（緩和版）
    private bool IsValidCenterForInterestingPoints(double centerX, double centerY, double zoom)
    {
        var range = 4.0 / zoom;
        var expandedRange = range * 3.0; // 3倍の範囲まで許可
        
        // より広い範囲で特徴点をチェック
        foreach (var point in InterestingPoints)
        {
            if (Math.Abs(point.X - centerX) <= expandedRange && 
                Math.Abs(point.Y - centerY) <= expandedRange)
            {
                return true;
            }
        }
        
        // 特徴点が完全に見えなくても、近い場合は許可
        var nearestDistance = InterestingPoints
            .Min(p => Math.Sqrt(Math.Pow(p.X - centerX, 2) + Math.Pow(p.Y - centerY, 2)));
        
        return nearestDistance <= range * 2.0; // 2倍の範囲内に最寄りの特徴点があれば許可
    }

    // 指定された位置に最も近い特徴点にナビゲート
    private void NavigateToNearestInterestingPoint(Point targetPosition)
    {
        var nearestPoint = InterestingPoints
            .OrderBy(p => Math.Sqrt(Math.Pow(p.X - targetPosition.X, 2) + Math.Pow(p.Y - targetPosition.Y, 2)))
            .First();
        
        // 最も近い特徴点に移動し、適切なズームレベルを設定
        _centerX = nearestPoint.X;
        _centerY = nearestPoint.Y;
        _zoom = Math.Min(_zoom, nearestPoint.OptimalZoom);
        
        UpdateInfo();
        _ = GenerateImageAsync();
    }

    private Complex ScreenToComplex(Point screenPoint)
    {
        // 表示サイズを取得
        var displayWidth = _actualImageWidth > 0 ? _actualImageWidth : 800;
        var displayHeight = _actualImageHeight > 0 ? _actualImageHeight : 800;

        // 正規化座標（-0.5～0.5）に変換
        var normalizedX = (screenPoint.X / displayWidth) - 0.5;
        var normalizedY = (screenPoint.Y / displayHeight) - 0.5; // Y軸は反転しない（統一性のため）

        // 複素平面の表示範囲を計算
        var scale = 4.0 / _zoom;
        var aspectRatio = displayWidth / displayHeight;
        
        // 複素平面座標に変換（アスペクト比考慮）
        var real = _centerX + normalizedX * scale * aspectRatio;
        var imag = _centerY + normalizedY * scale;

        return new Complex(real, imag);
    }

    private Point ComplexToScreen(Point complexPoint)
    {
        // 表示サイズを取得
        var displayWidth = _actualImageWidth > 0 ? _actualImageWidth : 800;
        var displayHeight = _actualImageHeight > 0 ? _actualImageHeight : 800;

        // 複素平面の表示範囲を計算
        var range = 4.0 / _zoom;
        
        // 正規化座標に変換
        var normalizedX = (complexPoint.X - _centerX) / range;
        var normalizedY = (complexPoint.Y - _centerY) / range;

        // 画面座標に変換
        var screenX = (normalizedX + 0.5) * displayWidth;
        var screenY = (0.5 - normalizedY) * displayHeight; // Y軸反転

        return new Point(screenX, screenY);
    }

    private void UpdateInfo()
    {
        var adaptiveIterations = CalculateAdaptiveIterations(_zoom);
        var displaySize = $"{_actualImageWidth:F0}×{_actualImageHeight:F0}";
        
        // 現在表示されている特徴点を検出
        var currentFeature = GetCurrentlyDisplayedFeature();
        var featureInfo = currentFeature != null ? $" - {currentFeature.Name}" : "";
        
        InfoLabel.Text = $"表示サイズ: {displaySize}, ズーム: {_zoom:E2}x, 中心: ({_centerX:F6}, {_centerY:F6}), 反復数: {adaptiveIterations}{featureInfo}";
        StatusLabel.Text = $"エンジン: {_mandelbrotService.GetEngineInfo()}, 解像度: {_imageWidth}×{_imageHeight}";
    }

    // 現在表示されている特徴点を取得
    private InterestingPoint? GetCurrentlyDisplayedFeature()
    {
        var range = 4.0 / _zoom;
        var threshold = range * 0.3; // 30%の範囲内にあれば「表示されている」とみなす
        
        return InterestingPoints
            .Where(p => Math.Abs(p.X - _centerX) <= threshold && Math.Abs(p.Y - _centerY) <= threshold)
            .OrderBy(p => Math.Sqrt(Math.Pow(p.X - _centerX, 2) + Math.Pow(p.Y - _centerY, 2)))
            .FirstOrDefault();
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

                    // 画面移動量を複素平面移動量に変換
                    var displayWidth = _actualImageWidth > 0 ? _actualImageWidth : 800;
                    var displayHeight = _actualImageHeight > 0 ? _actualImageHeight : 800;
                    var range = 4.0 / _zoom;

                    var complexDeltaX = -(deltaX / displayWidth) * range;
                    var complexDeltaY = (deltaY / displayHeight) * range; // Y軸反転

                    var newCenterX = _centerX + complexDeltaX;
                    var newCenterY = _centerY + complexDeltaY;
                    
                    // 新しい中心位置で特徴点が表示範囲内にあることを確認（緩和版）
                    if (IsValidCenterForInterestingPoints(newCenterX, newCenterY, _zoom))
                    {
                        _centerX = newCenterX;
                        _centerY = newCenterY;
                    }
                    else
                    {
                        // 制限されても部分的な移動は許可
                        var partialMoveX = complexDeltaX * 0.5; // 50%の移動を許可
                        var partialMoveY = complexDeltaY * 0.5;
                        
                        _centerX += partialMoveX;
                        _centerY += partialMoveY;
                    }
                    
                    _lastPanPoint = new Point(e.TotalX, e.TotalY);
                    UpdateInfo();
                }
                break;

            case GestureStatus.Completed:
                _lastPanPoint = null;
                _ = GenerateImageAsync(); // パン完亁E��に再生戁E
                break;
        }
    }

    private void OnPinchUpdated(object? sender, PinchGestureUpdatedEventArgs e)
    {
        if (_isGenerating) return;

        if (e.Status == GestureStatus.Running)
        {
            var newZoom = _zoom * e.Scale;
            
            // 特徴点が画面から外れないような最大ズームレベルを適用
            var maxAllowedZoom = CalculateMaxZoomForInterestingPoints();
            newZoom = Math.Max(0.1, Math.Min(maxAllowedZoom, newZoom));
            
            _zoom = newZoom;
            UpdateInfo();
        }
        else if (e.Status == GestureStatus.Completed)
        {
            _ = GenerateImageAsync(); // ピンチ完亁E��に再生戁E
        }
    }

    private async Task GenerateImageAsync()
    {
        if (_isGenerating) return;
        _isGenerating = true;

        try
        {
            // ズームレベルに応じて反復回数を�E動調整
            var adaptiveIterations = CalculateAdaptiveIterations(_zoom);
            
            MainThread.BeginInvokeOnMainThread(() =>
            {
                StatusLabel.Text = $"Computing Mandelbrot set... (Iterations: {adaptiveIterations})";
            });
            
            Debug.WriteLine($"Generating image: {_imageWidth}x{_imageHeight}, Center: ({_centerX}, {_centerY}), Zoom: {_zoom:E2}, Iterations: {adaptiveIterations}");
            
            // CUDA計算を実衁E
            var rgbaData = await _mandelbrotService.ComputeTileAsync(
                _centerX, _centerY, _zoom, 
                _imageWidth, _imageHeight, adaptiveIterations);
            
            Debug.WriteLine($"Computation completed. Data length: {rgbaData?.Length ?? 0}");
            
            if (rgbaData != null && rgbaData.Length > 0)
            {
                // RGBAチE�Eタの色刁E��
                AnalyzeColors(rgbaData);
                
                // ImageSourceを作�EしてImageコントロールに設宁E
                var imageSource = CreateImageSourceFromRgbaData(rgbaData, _imageWidth, _imageHeight);
                
                // UIスレチE��で画像を設宁E
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
        // 4K解像度に対応した高詳細設定
        var baseIterations = 256; // 基本反復数を増加
        var logZoom = Math.Log10(Math.Max(1.0, zoom));
        
        // より急激に反復数を増加させて詳細を確保
        var adaptiveIterations = (int)(baseIterations + logZoom * 100);
        
        // 高ズーム時のさらなる詳細化
        if (zoom > 1000)
        {
            adaptiveIterations += (int)((zoom / 1000) * 200);
        }
        
        // 最小512、最大8192で制限（4K解像度対応）
        return Math.Max(512, Math.Min(8192, adaptiveIterations));
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
        
        // 最初�E数ピクセルの詳細
        Debug.WriteLine("First 10 pixels:");
        for (int i = 0; i < Math.Min(10, rgbaData.Length / 4); i++)
        {
            var idx = i * 4;
            Debug.WriteLine($"Pixel {i}: RGBA({rgbaData[idx]}, {rgbaData[idx + 1]}, {rgbaData[idx + 2]}, {rgbaData[idx + 3]})");
        }
    }

    private ImageSource? CreateImageSourceFromRgbaData(byte[] rgbaData, int width, int height)
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

    // 興味深ぁE��所への移動機�E
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
