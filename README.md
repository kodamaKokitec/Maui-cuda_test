# CUDA Mandelbrot Set Visualization

このプロジェクトは、CUDA（Compute Unified Device Architecture）を使用してMandelbrot集合を可視化するテストプログラムです。VRAMにデータを移動し、GPU上で並列演算を行い、視覚的な結果を生成します。

## 🚀 機能

### 高性能CUDAエンジン
- CUDA 12.4を使用した並列Mandelbrot集合計算
- RTX 4060 Tiで1294.56 Mpixels/secの高速処理
- HSV色空間による鮮やかな色彩表現
- 対数スケーリングによる境界構造の詳細可視化

### .NET MAUI インタラクティブアプリケーション
- 1024x1024高解像度レンダリング
- 直感的なマウス操作：
  - 左クリック: 2倍ズームイン
  - 右クリック: 0.5倍ズームアウト  
  - ダブルクリック: 4倍ズームイン
  - ドラッグ: パン操作
- 修正された座標計算システム（パン後のズーム座標ずれを解決）
- 適応的反復計算による詳細境界探索

## 🛠️ 必要環境

- Windows 10/11
- NVIDIA GPU（CUDA対応、RTX 4060 Ti推奨）
- CUDA Toolkit 12.x以降
- Visual Studio 2022（C++コンパイラ）
- .NET 10.0 SDK
- CUDA対応GPU（Compute Capability 3.5以降推奨）

## 📁 プロジェクト構成

### CUDAコア
- `mandelbrot_cuda_clean.cu`: スタンドアロンCUDAベースライン
- `MandelbrotCudaWrapper.cu`: P/Invokeブリッジ（高度な色彩アルゴリズム）
- `hello_cuda.cu`: 基本CUDAテストプログラム

### .NET MAUIアプリケーション
- `MandelbrotMAUI/`: メインアプリケーション
- `MainPage_Image.xaml/.cs`: 修正された座標変換とジェスチャー処理
- `CudaMandelbrotService.cs`: CUDA統合サービス層

### テストプログラム
- `test_rgb_values.cpp`: 色彩出力検証
- `test_cuda_wrapper_bmp.cpp`: BMPファイル生成テスト
- `test_debug_params.cpp`: デバッグパラメータ検証

## 🏗️ ビルド方法

### CUDAエンジン
```bash
# CUDAラッパーDLLをビルド
build_cuda_wrapper_vs2022.bat

# スタンドアロンCUDAプログラム
build_clean.bat
```

### MAUIアプリケーション
```bash
cd MandelbrotMAUI
dotnet build
dotnet run --framework net10.0-windows10.0.19041.0
```

## 🎯 実行方法

### スタンドアロン実行
```bash
# 基本テスト
.\hello_cuda.exe

# 高解像度Mandelbrot生成
.\mandelbrot_cuda_clean.exe
```

### インタラクティブアプリ
```bash
cd MandelbrotMAUI
dotnet run --framework net10.0-windows10.0.19041.0
```

## 📊 パフォーマンス

**RTX 4060 Ti での実測値:**
- 処理速度: 1294.56 Mpixels/sec
- レンダリング時間: < 1ms (1024x1024)
- 色彩分布: Black: 25,395, Blue: 1,252, Cyan: 3,823, Green: 7,882, Yellow: 88,314, Red: 71,391

## 🎨 出力

- `mandelbrot.bmp`: 高解像度Mandelbrot可視化（Windows標準対応）
- リアルタイム MAUI UI: インタラクティブ探索
- コンソール出力: GPU情報、実行時間、パフォーマンス統計

## 🔧 技術的詳細

### CUDA仕様
- 画像サイズ: 1024x1024ピクセル
- 適応的反復回数: 1000-10000回（ズームレベル依存）
- ブロックサイズ: 16x16スレッド
- メモリ使用量: 約12MB（GPU側）

### 色彩アルゴリズム
- HSV色空間マッピング
- 対数スケーリング
- RGB→RGBA変換（UI統合用）

### 座標システム修正
- 正確なスクリーン→複素座標変換
- アスペクト比考慮
- ズーム中心位置維持
- パン操作の座標ドリフト解決

## 🐛 解決済み問題

- ✅ 色表示問題（赤単色→多彩な色彩）
- ✅ GraphicsView不安定性→Image制御移行
- ✅ 座標計算精度→数学的変換修正
- ✅ パン後ズーム座標ずれ→ZoomAtPosition実装
- ✅ マウス操作直感性→標準的な操作系統一

## 📋 使用方法

1. **基本探索**: アプリ起動後、左クリックで興味のある領域をズーム
2. **詳細観察**: 境界付近でのダブルクリックで微細構造を観察
3. **ナビゲーション**: ドラッグでパン、右クリックでズームアウト
4. **高速移動**: 画面上部の興味深い位置リストから直接ジャンプ

## 🔬 検証済みテスト

- CUDA計算精度確認済み
- 色彩アルゴリズム多様性確認済み
- Image制御安定性確認済み
- 座標計算数学的正確性確認済み
- マウス操作直感性確認済み