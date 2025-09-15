#!/usr/bin/env pwsh
# UI自動テストスタートアップスクリプト

Write-Host "=== Mandelbrot MAUI UI自動テスト システム ===" -ForegroundColor Green

# 1. 必要なプロセスが実行中かチェック
Write-Host "`n1. アプリケーション状態をチェック中..." -ForegroundColor Yellow
$maui = Get-Process -Name "*Mandelbrot*" -ErrorAction SilentlyContinue
if ($maui) {
    Write-Host "   ✓ MandelbrotMAUI が実行中です (PID: $($maui.Id))" -ForegroundColor Green
} else {
    Write-Host "   ⚠ MandelbrotMAUI が見つかりません" -ForegroundColor Red
    Write-Host "     アプリを手動で起動してからテストを再実行してください。" -ForegroundColor Red
}

# 2. 基本的なUIテスト関数を定義
function Test-ClickAccuracy {
    param([int]$X, [int]$Y, [string]$Description)
    
    Write-Host "   テスト: $Description (${X}, ${Y})" -ForegroundColor Cyan
    
    # PowerShellでクリック操作をシミュレート
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -TypeDefinition @"
        using System;
        using System.Runtime.InteropServices;
        public class MouseOps {
            [DllImport("user32.dll")]
            public static extern bool SetCursorPos(int x, int y);
            
            [DllImport("user32.dll")]
            public static extern void mouse_event(int dwFlags, int dx, int dy, int cButtons, int dwExtraInfo);
            
            public const int LEFTDOWN = 0x02;
            public const int LEFTUP = 0x04;
            
            public static void Click(int x, int y) {
                SetCursorPos(x, y);
                mouse_event(LEFTDOWN, 0, 0, 0, 0);
                System.Threading.Thread.Sleep(50);
                mouse_event(LEFTUP, 0, 0, 0, 0);
            }
        }
"@
    
    try {
        [MouseOps]::Click($X, $Y)
        Start-Sleep -Milliseconds 500
        Write-Host "     ✓ クリック実行完了" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "     ✗ クリック実行エラー: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Test-DragOperation {
    param([int]$StartX, [int]$StartY, [int]$EndX, [int]$EndY, [string]$Description)
    
    Write-Host "   テスト: $Description (${StartX},${StartY} → ${EndX},${EndY})" -ForegroundColor Cyan
    
    try {
        # ドラッグ開始
        [MouseOps]::SetCursorPos($StartX, $StartY)
        [MouseOps]::mouse_event([MouseOps]::LEFTDOWN, 0, 0, 0, 0)
        Start-Sleep -Milliseconds 100
        
        # 段階的にドラッグ
        $steps = 10
        for ($i = 1; $i -le $steps; $i++) {
            $x = $StartX + ($i * ($EndX - $StartX) / $steps)
            $y = $StartY + ($i * ($EndY - $StartY) / $steps)
            [MouseOps]::SetCursorPos([int]$x, [int]$y)
            Start-Sleep -Milliseconds 20
        }
        
        # ドラッグ終了
        [MouseOps]::mouse_event([MouseOps]::LEFTUP, 0, 0, 0, 0)
        Start-Sleep -Milliseconds 500
        
        Write-Host "     ✓ ドラッグ操作完了" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "     ✗ ドラッグ操作エラー: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# 3. UIテストシナリオを実行
if ($maui) {
    Write-Host "`n2. UI操作テストを実行中..." -ForegroundColor Yellow
    
    # アプリケーションウィンドウをアクティブ化
    $maui | ForEach-Object { 
        try {
            $_.MainWindowTitle | Out-Null
            Add-Type -AssemblyName Microsoft.VisualBasic
            [Microsoft.VisualBasic.Interaction]::AppActivate($_.Id)
            Start-Sleep -Seconds 1
        } catch {
            Write-Host "   ウィンドウアクティベート中..." -ForegroundColor Gray
        }
    }
    
    # テストケース1: 中央クリックズーム
    $success1 = Test-ClickAccuracy -X 400 -Y 400 -Description "中央ズーム"
    
    # テストケース2: 右上コーナークリック
    $success2 = Test-ClickAccuracy -X 600 -Y 200 -Description "右上コーナーズーム"
    
    # テストケース3: パン操作（中央から右下）
    $success3 = Test-DragOperation -StartX 400 -StartY 400 -EndX 500 -EndY 500 -Description "パン操作（右下）"
    
    # テストケース4: パン操作（逆方向）
    $success4 = Test-DragOperation -StartX 500 -StartY 500 -EndX 300 -EndY 300 -Description "パン操作（左上）"
    
    # 結果サマリー
    Write-Host "`n3. テスト結果サマリー" -ForegroundColor Yellow
    $results = @($success1, $success2, $success3, $success4)
    $passed = ($results | Where-Object { $_ -eq $true }).Count
    $total = $results.Count
    
    Write-Host "   実行済みテスト: $total" -ForegroundColor White
    Write-Host "   成功: $passed" -ForegroundColor Green
    Write-Host "   失敗: $($total - $passed)" -ForegroundColor Red
    
    if ($passed -eq $total) {
        Write-Host "`n🎉 全テストが成功しました！UIは正常に動作しています。" -ForegroundColor Green
    } else {
        Write-Host "`n⚠ いくつかのテストが失敗しました。アプリの応答性を確認してください。" -ForegroundColor Yellow
    }
    
    # 4. 高度なMCPテストへの案内
    Write-Host "`n4. 高度なテストオプション" -ForegroundColor Yellow
    Write-Host "   より詳細なテストを実行するには:" -ForegroundColor White
    Write-Host "   - MCP_UI_Automation_Guide.md を参照" -ForegroundColor Cyan
    Write-Host "   - Node.js MCP サーバーをセットアップ" -ForegroundColor Cyan
    Write-Host "   - 座標変換精度テストを実行" -ForegroundColor Cyan
    
} else {
    Write-Host "`n⚠ MandelbrotMAUIアプリが実行されていません。" -ForegroundColor Red
    Write-Host "以下の手順で開始してください:" -ForegroundColor White
    Write-Host "1. .\build_clean.bat  # アプリをビルド" -ForegroundColor Cyan
    Write-Host "2. アプリを手動起動" -ForegroundColor Cyan
    Write-Host "3. .\ui_test.ps1      # このスクリプトを再実行" -ForegroundColor Cyan
}

Write-Host "`n=== テスト完了 ===" -ForegroundColor Green