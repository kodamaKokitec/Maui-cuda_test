using System;

namespace MandelbrotCoordinateTest
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("=== マンデルブロー座標変換テスト ===");
            
            var testHelper = new CoordinateTestHelper();
            
            // テストケース1: 画面中央 (400, 400)
            Console.WriteLine("\n1. 画面中央テスト:");
            TestCoordinate(testHelper, 400, 400, -0.5, 0.0, 1.0, 800, 800);
            
            // テストケース2: 左上角 (200, 200)
            Console.WriteLine("\n2. 左上角テスト:");
            TestCoordinate(testHelper, 200, 200, -0.5, 0.0, 1.0, 800, 800);
            
            // テストケース3: 右下角 (600, 600)
            Console.WriteLine("\n3. 右下角テスト:");
            TestCoordinate(testHelper, 600, 600, -0.5, 0.0, 1.0, 800, 800);
            
            // テストケース4: ズーム2倍での中央
            Console.WriteLine("\n4. ズーム2倍での中央テスト:");
            TestCoordinate(testHelper, 400, 400, -0.5, 0.0, 2.0, 800, 800);
            
            Console.WriteLine("\n=== テスト完了 ===");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
        
        static void TestCoordinate(CoordinateTestHelper helper, double screenX, double screenY, 
            double centerX, double centerY, double zoom, double displayWidth, double displayHeight)
        {
            var screenPoint = new TestPoint(screenX, screenY);
            var complexPoint = helper.ScreenToComplex(screenPoint, centerX, centerY, zoom, displayWidth, displayHeight);
            var backToScreen = helper.ComplexToScreen(complexPoint, centerX, centerY, zoom, displayWidth, displayHeight);
            
            var errorX = Math.Abs(screenX - backToScreen.X);
            var errorY = Math.Abs(screenY - backToScreen.Y);
            
            Console.WriteLine($"  画面座標: ({screenX:F1}, {screenY:F1})");
            Console.WriteLine($"  複素座標: ({complexPoint.X:F6}, {complexPoint.Y:F6})");
            Console.WriteLine($"  逆変換:   ({backToScreen.X:F1}, {backToScreen.Y:F1})");
            Console.WriteLine($"  誤差:     X={errorX:F3}, Y={errorY:F3}");
            
            if (errorX <= 1.0 && errorY <= 1.0)
            {
                Console.WriteLine("  結果: ✓ 合格");
            }
            else
            {
                Console.WriteLine("  結果: ✗ 不合格");
            }
        }
    }

    public class CoordinateTestHelper
    {
        public TestPoint ScreenToComplex(TestPoint screenPoint, double centerX, double centerY, double zoom, double displayWidth, double displayHeight)
        {
            // 正規化座標（-0.5～0.5）に変換
            var normalizedX = (screenPoint.X / displayWidth) - 0.5;
            var normalizedY = 0.5 - (screenPoint.Y / displayHeight); // Y軸反転

            // 複素平面の表示範囲を計算
            var range = 4.0 / zoom;
            
            // 複素平面座標に変換
            var complexX = centerX + normalizedX * range;
            var complexY = centerY + normalizedY * range;

            return new TestPoint(complexX, complexY);
        }

        public TestPoint ComplexToScreen(TestPoint complexPoint, double centerX, double centerY, double zoom, double displayWidth, double displayHeight)
        {
            // 複素平面の表示範囲を計算
            var range = 4.0 / zoom;
            
            // 正規化座標に変換
            var normalizedX = (complexPoint.X - centerX) / range;
            var normalizedY = (complexPoint.Y - centerY) / range;

            // 画面座標に変換
            var screenX = (normalizedX + 0.5) * displayWidth;
            var screenY = (0.5 - normalizedY) * displayHeight; // Y軸反転

            return new TestPoint(screenX, screenY);
        }
    }

    public struct TestPoint
    {
        public double X { get; set; }
        public double Y { get; set; }

        public TestPoint(double x, double y)
        {
            X = x;
            Y = y;
        }
    }
}