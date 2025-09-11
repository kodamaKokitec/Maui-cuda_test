# CUDA Mandelbrot Set Visualization

このプロジェクトは、CUDA（Compute Unified Device Architecture）を使用してMandelbrot集合を可視化するテストプログラムです。VRAMにデータを移動し、GPU上で並列演算を行い、視覚的な結果を生成します。

## 機能

- CUDA を使用した並列Mandelbrot集合計算
- VRAMへのデータ転送とGPU演算
- PPMフォーマットでの画像出力
- パフォーマンス測定機能

## 必要環境

- Windows 10/11
- NVIDIA GPU（CUDA対応）
- CUDA Toolkit 12.x以降
- Visual Studio 2019/2022（C++コンパイラ）
- CUDA対応GPU（Compute Capability 3.5以降推奨）

## ビルド方法

```bash
# Visual Studio Developer Command Promptで実行
nvcc -o mandelbrot_cuda mandelbrot_cuda.cu
```

## 実行方法

```bash
# 実行ファイルを起動
./mandelbrot_cuda
```

生成された `mandelbrot.ppm` ファイルを画像ビューアで開いて結果を確認できます。

## 出力

- `mandelbrot.bmp`: Mandelbrot集合の可視化画像（Windows標準対応）
- `mandelbrot.ppm`: Mandelbrot集合の可視化画像（Portable Pixmap形式）
- コンソール出力: GPU情報、実行時間、パフォーマンス統計

## 画像の確認方法

**Windows標準の方法:**

- `mandelbrot.bmp`をダブルクリックしてWindowsフォトビューアーで開く

**その他の画像ビューア:**

- IrfanView, GIMP, Photoshop等で両形式を開ける

## 技術的詳細

- 画像サイズ: 1024x1024ピクセル
- 最大反復回数: 1000回
- ブロックサイズ: 16x16スレッド
- メモリ使用量: 約12MB（GPU側）

## ファイル構成

- `mandelbrot_cuda_clean.cu`: メインのCUDAプログラム
- `hello_cuda.cu`: 基本的なCUDAテストプログラム
- `build_clean.bat`: コンパイル用バッチファイル
- `README.md`: このファイル
- `mandelbrot.bmp`: 生成される出力画像（Windows標準形式）
- `mandelbrot.ppm`: 生成される出力画像（PPM形式）
