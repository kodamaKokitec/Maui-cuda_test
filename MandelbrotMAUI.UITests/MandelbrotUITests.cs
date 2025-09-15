using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Drawing;

namespace MandelbrotMAUI.UITests
{
    [TestClass]
    public class MandelbrotUITests
    {
        private ILogger<MandelbrotUITests> _logger;

        [TestInitialize]
        public void Setup()
        {
            var loggerFactory = LoggerFactory.Create(builder =>
                builder.AddConsole().SetMinimumLevel(LogLevel.Debug));
            _logger = loggerFactory.CreateLogger<MandelbrotUITests>();
        }

        [TestMethod]
        public void TestCoordinateTransformation()
        {
            _logger.LogInformation("Testing coordinate transformation accuracy...");
            
            // MainPage_Imageのインスタンスを作成（テスト用）
            var testHelper = new CoordinateTestHelper();
            
            // テストケース1: 画面中央
            var screenCenter = new Point(400, 400); // 800x800の中央
            var complexCenter = testHelper.ScreenToComplex(screenCenter, -0.5, 0.0, 1.0, 800, 800);
            var backToScreen = testHelper.ComplexToScreen(complexCenter, -0.5, 0.0, 1.0, 800, 800);
            
            _logger.LogInformation($"Screen center: ({screenCenter.X}, {screenCenter.Y})");
            _logger.LogInformation($"Complex center: ({complexCenter.X:F6}, {complexCenter.Y:F6})");
            _logger.LogInformation($"Back to screen: ({backToScreen.X:F1}, {backToScreen.Y:F1})");
            
            // 誤差1ピクセル以内であることを確認
            Assert.IsTrue(Math.Abs(screenCenter.X - backToScreen.X) <= 1.0, 
                "X coordinate transformation roundtrip failed");
            Assert.IsTrue(Math.Abs(screenCenter.Y - backToScreen.Y) <= 1.0, 
                "Y coordinate transformation roundtrip failed");
        }

        [TestMethod]
        public void TestZoomFunctionality()
        {
            _logger.LogInformation("Testing zoom functionality...");
            
            var testHelper = new CoordinateTestHelper();
            
            // テストケース: 左上角をクリックしてズーム
            var clickPoint = new Point(200, 200); // 左上寄り
            var initialCenter = new Point(-0.5, 0.0);
            var initialZoom = 1.0;
            
            // クリック点の複素座標を計算
            var clickComplex = testHelper.ScreenToComplex(clickPoint, initialCenter.X, initialCenter.Y, initialZoom, 800, 800);
            _logger.LogInformation($"Click point: ({clickPoint.X}, {clickPoint.Y})");
            _logger.LogInformation($"Click complex: ({clickComplex.X:F6}, {clickComplex.Y:F6})");
            
            // ズーム後、クリック点が新しい中心になることを確認
            var newZoom = initialZoom * 2.0;
            var expectedNewCenter = clickComplex;
            
            // 新しい中心で画面中央に戻る座標を計算
            var screenCenterAfterZoom = testHelper.ComplexToScreen(expectedNewCenter, expectedNewCenter.X, expectedNewCenter.Y, newZoom, 800, 800);
            
            _logger.LogInformation($"Expected new center: ({expectedNewCenter.X:F6}, {expectedNewCenter.Y:F6})");
            _logger.LogInformation($"Screen center after zoom: ({screenCenterAfterZoom.X:F1}, {screenCenterAfterZoom.Y:F1})");
            
            // 画面中央（400, 400）に近いことを確認
            Assert.IsTrue(Math.Abs(400 - screenCenterAfterZoom.X) <= 5.0, 
                "Zoom center X calculation failed");
            Assert.IsTrue(Math.Abs(400 - screenCenterAfterZoom.Y) <= 5.0, 
                "Zoom center Y calculation failed");
        }

        [TestMethod]
        public void TestPanFunctionality()
        {
            _logger.LogInformation("Testing pan functionality...");
            
            var testHelper = new CoordinateTestHelper();
            
            // 初期状態
            var initialCenterX = -0.5;
            var initialCenterY = 0.0;
            var zoom = 1.0;
            
            // 100ピクセル右に移動
            var deltaX = 100.0;
            var deltaY = 0.0;
            
            var range = 4.0 / zoom;
            var complexDeltaX = -(deltaX / 800.0) * range;
            var complexDeltaY = (deltaY / 800.0) * range;
            
            var newCenterX = initialCenterX + complexDeltaX;
            var newCenterY = initialCenterY + complexDeltaY;
            
            _logger.LogInformation($"Pan delta: ({deltaX}, {deltaY}) pixels");
            _logger.LogInformation($"Complex delta: ({complexDeltaX:F6}, {complexDeltaY:F6})");
            _logger.LogInformation($"New center: ({newCenterX:F6}, {newCenterY:F6})");
            
            // パン後の座標が期待値と一致することを確認
            var expectedDelta = -(100.0 / 800.0) * 4.0; // -0.5
            Assert.IsTrue(Math.Abs(complexDeltaX - expectedDelta) <= 0.01, 
                "Pan calculation failed");
        }
    }

    // テスト用ヘルパークラス
    public class CoordinateTestHelper
    {
        public Point ScreenToComplex(Point screenPoint, double centerX, double centerY, double zoom, double displayWidth, double displayHeight)
        {
            // 正規化座標（-0.5～0.5）に変換
            var normalizedX = (screenPoint.X / displayWidth) - 0.5;
            var normalizedY = 0.5 - (screenPoint.Y / displayHeight); // Y軸反転

            // 複素平面の表示範囲を計算
            var range = 4.0 / zoom;
            
            // 複素平面座標に変換
            var complexX = centerX + normalizedX * range;
            var complexY = centerY + normalizedY * range;

            return new Point(complexX, complexY);
        }

        public Point ComplexToScreen(Point complexPoint, double centerX, double centerY, double zoom, double displayWidth, double displayHeight)
        {
            // 複素平面の表示範囲を計算
            var range = 4.0 / zoom;
            
            // 正規化座標に変換
            var normalizedX = (complexPoint.X - centerX) / range;
            var normalizedY = (complexPoint.Y - centerY) / range;

            // 画面座標に変換
            var screenX = (normalizedX + 0.5) * displayWidth;
            var screenY = (0.5 - normalizedY) * displayHeight; // Y軸反転

            return new Point(screenX, screenY);
        }
    }

    // テスト用のPoint構造体
    public struct Point
    {
        public double X { get; set; }
        public double Y { get; set; }

        public Point(double x, double y)
        {
            X = x;
            Y = y;
        }
    }
}