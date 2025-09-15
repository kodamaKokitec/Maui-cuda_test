# Visual Studio Debug Configuration for MAUI CUDA Project

## プロジェクト構成

### ソリューション構造
- **MandelbrotMAUI** (.NET 10.0 MAUI): メインアプリケーション (C#)
- **MandelbrotCudaWrapper** (Visual C++): CUDA DLL プロジェクト (C++/CUDA)

### デバッグ設定

#### MAUI アプリケーションのデバッグ

1. **Visual Studio でソリューションを開く**
   ```
   devenv C:\Temp\cuda\cuda.sln
   ```

2. **デバッグ構成を選択**
   - Configuration: `Debug`
   - Platform: `x64`
   - Startup Project: `MandelbrotMAUI`

3. **デバッグプロファイル**
   - **Windows Machine**: 標準デバッグ (マネージドコードのみ)
   - **Windows Machine (Native Debug)**: ネイティブコードデバッグ有効

#### CUDA コードのデバッグ

1. **CUDAプロジェクトの設定**
   - CUDA Toolkit 12.4 が必要
   - Compute Capability: 5.0, 6.1, 7.5, 8.6, 8.9 をサポート
   - デバッグシンボル生成: 有効

2. **混合モードデバッグ**
   - MAUIプロジェクトのプロパティ → デバッグ → ネイティブコードデバッグを有効
   - ステップ実行で C# から CUDA コードまで追跡可能

#### デバッグ機能

1. **コンソール出力**
   ```csharp
   CudaDebugHelper.EnableConsoleOutput(); // デバッグ用コンソール表示
   ```

2. **パフォーマンス計測**
   ```csharp
   CudaDebugHelper.LogPerformance("Operation", elapsed, pixelCount);
   ```

3. **ログファイル**
   - `cuda_debug.log`: 実行時ログ
   - Visual Studio 出力ウィンドウにもログ表示

### ビルド手順

#### Visual Studio での実行
1. ソリューションを開く
2. スタートアッププロジェクトを `MandelbrotMAUI` に設定
3. `F5` でデバッグ実行

#### コマンドラインでの実行
```bash
# MAUIのみ (推奨)
dotnet build MandelbrotMAUI\MandelbrotMAUI.csproj
dotnet run --project MandelbrotMAUI\MandelbrotMAUI.csproj

# CUDAプロジェクトはVisual Studioでのみビルド可能
```

### トラブルシューティング

#### CUDA DLL が見つからない場合
- `MandelbrotMAUI\Native\MandelbrotCudaWrapper.dll` の存在確認
- プロジェクト依存関係の設定確認

#### デバッグシンボルが読み込まれない場合
- CUDAプロジェクトでデバッグ情報生成を確認
- 両プロジェクトの出力フォルダが同じであることを確認

#### パフォーマンス問題
- CUDAデバッガーのオーバーヘッドを考慮
- リリースビルドでの性能測定を実施

### 推奨開発環境
- Visual Studio 2022 Professional 以上
- CUDA Toolkit 12.4
- Windows 10/11 (x64)
- NVIDIA GPU (Compute Capability 5.0以上)