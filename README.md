# CUDA Mandelbrot Set Visualization

このプロジェクトは、CUDA（Compute Unified Device Architecture）を使用してMandelbrot集合を可視化するテストプログラムです。VRAMにデータを移動し、GPU上で並列演算を行い、視覚的な結果を生成します。

## 🚀 機能

### 高性能CUDAエンジン
- CUDA 12.4を使用した並列Mandelbrot集合計算
- RTX 4060 Tiで11.04 Mpixels/secの高速処理
- RGB色空間による鮮やかな色彩表現
- 詳細な性能統計とGPU情報表示

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
- Visual Studio 2019/2022（C++コンパイラ）
- .NET 10.0 SDK（MAUIアプリ用）
- CUDA対応GPU（Compute Capability 3.5以降推奨）

## 📁 プロジェクト構成

### CUDAコア
- `mandelbrot_cuda_clean_fixed.cu`: 修正済みMandelbrotCUDAプログラム
- `hello_cuda_new.cu`: 基本CUDAテストプログラム
- `MandelbrotCudaWrapper.cu`: P/Invokeブリッジ（高度な色彩アルゴリズム）

### .NET MAUIアプリケーション
- `MandelbrotMAUI/`: メインアプリケーション
- `MainPage_Image.xaml/.cs`: 修正された座標変換とジェスチャー処理
- `CudaMandelbrotService.cs`: CUDA統合サービス層

### テストプログラム
- `test_cuda_wrapper.cpp`: CUDAラッパーDLLテスト
- `test_rgb_values.cpp`: 色彩出力検証
- `test_cuda_wrapper_bmp.cpp`: BMPファイル生成テスト

## 🏗️ ビルド方法

### 自動環境検出ビルド（推奨）

```bash
# 全プログラムを自動ビルド（環境自動検出）
build_all.bat

# デバッグビルド
build_debug.bat --debug

# リリースビルド  
build_debug.bat --release

# 詳細出力付きビルド
build_debug.bat --debug --verbose
```

**ビルドシステムの特徴:**
- Visual Studio 2019/2022の自動検出
- CUDA Toolkit の自動検出
- x64アーキテクチャ対応
- デバッグシンボル生成
- エラーハンドリング完備

### 手動ビルド

```bash
# 個別ビルド（Visual Studio環境設定後）
nvcc -o hello_cuda_new.exe hello_cuda_new.cu
nvcc -o mandelbrot_cuda_clean_fixed.exe mandelbrot_cuda_clean_fixed.cu
cl /EHsc test_cuda_wrapper.cpp /Fe:test_cuda_wrapper_x64.exe
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
# 基本CUDAテスト
.\hello_cuda_new.exe

# 高解像度Mandelbrot生成（BMPとPPM形式）
.\mandelbrot_cuda_clean_fixed.exe

# CUDAラッパーテスト
.\test_cuda_wrapper_x64.exe
```

### デバッグ実行

```bash
# デバッグビルド実行
.\hello_cuda_new_DEBUG.exe
.\mandelbrot_cuda_clean_fixed_DEBUG.exe
.\test_cuda_wrapper_DEBUG.exe
```

### インタラクティブアプリ

```bash
cd MandelbrotMAUI
dotnet run --framework net10.0-windows10.0.19041.0
```

## 📊 パフォーマンス

**RTX 4060 Ti での実測値:**
- **Mandelbrot計算**: 11.04 Mpixels/sec
- **メモリスループット**: 31.58 MB/s
- **実行時間**: 95ms (1024x1024ピクセル)
- **GPU使用率**: 高効率並列処理

## 🖼️ 出力ファイル

- `mandelbrot.bmp`: Mandelbrot集合の可視化画像（Windows標準対応）
- `mandelbrot.ppm`: Mandelbrot集合の可視化画像（Portable Pixmap形式）
- コンソール出力: GPU情報、実行時間、パフォーマンス統計

## 🐛 デバッグ

### デバッグビルドの使用

```bash
# デバッグシンボル付きビルド
build_debug.bat --debug --verbose

# Visual Studio でデバッグ
# 1. Visual Studio でプロジェクトを開く
# 2. デバッグ実行ファイルを設定
# 3. ブレークポイントを設定して実行
```

### サポートするデバッグツール

- **Visual Studio Debugger**: C++コードのデバッグ
- **NVIDIA Nsight Compute**: CUDAカーネルプロファイリング
- **CUDA-GDB**: コマンドラインCUDAデバッグ

## 🔧 トラブルシューティング

### よくある問題

1. **CUDA Toolkit が見つからない**
   ```
   解決策: CUDA Toolkit 12.x以降をインストール
   https://developer.nvidia.com/cuda-downloads
   ```

2. **Visual Studio が見つからない**
   ```
   解決策: Visual Studio 2019/2022 with C++ tools をインストール
   ```

3. **アーキテクチャエラー**
   ```
   解決策: x64環境でビルド（build_all.batが自動設定）
   ```

## 🚀 技術的詳細

- **画像サイズ**: 1024x1024ピクセル
- **最大反復回数**: 1000回
- **ブロックサイズ**: 16x16スレッド
- **グリッドサイズ**: 64x64ブロック
- **メモリ使用量**: 約3MB（GPU側）
- **色空間**: RGB 24-bit

## 📄 ライセンス

このプロジェクトはオープンソースです。詳細はLICENSEファイルを参照してください。

## 🤝 貢献

プルリクエストや課題報告を歓迎します。貢献ガイドラインについては、CONTRIBUTINGファイルを参照してください。