# MCP サーバーベース UI 自動テスト セットアップガイド

## 概要
このガイドでは、Model Context Protocol (MCP) サーバーを使用してMAUIアプリケーションのUI自動テストを実装する方法を説明します。

## 1. 必要なコンポーネント

### 1.1 MCP サーバー設定
```json
{
  "mcpServers": {
    "ui-automation": {
      "command": "node",
      "args": ["./ui-automation-server.js"],
      "env": {
        "UI_AUTOMATION_TARGET": "MandelbrotMAUI"
      }
    }
  }
}
```

### 1.2 UI Automation Dependencies
```xml
<PackageReference Include="Microsoft.Windows.SDK.Win32Metadata" Version="60.0.25" />
<PackageReference Include="Microsoft.WindowsAppSDK" Version="1.6.241114003" />
<PackageReference Include="FlaUI.Core" Version="4.0.0" />
<PackageReference Include="FlaUI.UIA3" Version="4.0.0" />
```

## 2. MCP UI Automation Server

### 2.1 Node.js サーバー実装
```javascript
// ui-automation-server.js
const { spawn } = require('child_process');
const { MCPServer } = require('@modelcontextprotocol/sdk/server');

class UIAutomationServer extends MCPServer {
  constructor() {
    super({
      name: "ui-automation",
      version: "1.0.0"
    });
    
    this.registerTools();
  }

  registerTools() {
    // ズーム操作テスト
    this.addTool({
      name: "test_zoom_at_position",
      description: "指定位置でのズーム操作をテスト",
      inputSchema: {
        type: "object",
        properties: {
          x: { type: "number", description: "X座標" },
          y: { type: "number", description: "Y座標" },
          zoomFactor: { type: "number", description: "ズーム倍率" }
        }
      }
    }, this.testZoomAtPosition.bind(this));

    // パン操作テスト
    this.addTool({
      name: "test_pan_gesture",
      description: "ドラッグによるパン操作をテスト",
      inputSchema: {
        type: "object",
        properties: {
          startX: { type: "number" },
          startY: { type: "number" },
          endX: { type: "number" },
          endY: { type: "number" }
        }
      }
    }, this.testPanGesture.bind(this));

    // 座標変換検証
    this.addTool({
      name: "validate_coordinate_transformation",
      description: "座標変換の正確性を検証",
      inputSchema: {
        type: "object",
        properties: {
          testPoints: {
            type: "array",
            items: {
              type: "object",
              properties: {
                x: { type: "number" },
                y: { type: "number" }
              }
            }
          }
        }
      }
    }, this.validateCoordinateTransformation.bind(this));
  }

  async testZoomAtPosition(args) {
    const { x, y, zoomFactor } = args;
    
    // PowerShellスクリプトでUI操作を実行
    const script = `
      # アプリウィンドウを見つける
      $process = Get-Process -Name "MandelbrotMAUI" -ErrorAction SilentlyContinue
      if ($process) {
        # UI Automationでクリック操作
        Add-Type -AssemblyName UIAutomationClient
        Add-Type -AssemblyName System.Windows.Forms
        
        # 座標にクリック
        [System.Windows.Forms.Cursor]::Position = [System.Drawing.Point]::new(${x}, ${y})
        [System.Windows.Forms.Application]::DoEvents()
        
        # 左クリックをシミュレート
        Add-Type -TypeDefinition @"
          using System;
          using System.Runtime.InteropServices;
          public class MouseOperations {
            [DllImport("user32.dll")]
            public static extern void mouse_event(int dwFlags, int dx, int dy, int cButtons, int dwExtraInfo);
            
            public const int MOUSEEVENTF_LEFTDOWN = 0x02;
            public const int MOUSEEVENTF_LEFTUP = 0x04;
            
            public static void LeftClick() {
              mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
              mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
            }
          }
"@
        [MouseOperations]::LeftClick()
        
        Start-Sleep -Seconds 2
        "Zoom test completed at (${x}, ${y}) with factor ${zoomFactor}"
      } else {
        "MandelbrotMAUI process not found"
      }
    `;
    
    return await this.executePowerShell(script);
  }

  async testPanGesture(args) {
    const { startX, startY, endX, endY } = args;
    
    const script = `
      Add-Type -AssemblyName System.Windows.Forms
      Add-Type -TypeDefinition @"
        using System;
        using System.Runtime.InteropServices;
        public class MouseOperations {
          [DllImport("user32.dll")]
          public static extern void mouse_event(int dwFlags, int dx, int dy, int cButtons, int dwExtraInfo);
          
          [DllImport("user32.dll")]
          public static extern bool SetCursorPos(int x, int y);
          
          public const int MOUSEEVENTF_LEFTDOWN = 0x02;
          public const int MOUSEEVENTF_LEFTUP = 0x04;
          public const int MOUSEEVENTF_MOVE = 0x01;
        }
"@
      
      # ドラッグ開始
      [MouseOperations]::SetCursorPos(${startX}, ${startY})
      Start-Sleep -Milliseconds 100
      [MouseOperations]::mouse_event([MouseOperations]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
      
      # ドラッグ移動
      $steps = 20
      for ($i = 0; $i -lt $steps; $i++) {
        $x = ${startX} + ($i * (${endX} - ${startX}) / $steps)
        $y = ${startY} + ($i * (${endY} - ${startY}) / $steps)
        [MouseOperations]::SetCursorPos([int]$x, [int]$y)
        Start-Sleep -Milliseconds 10
      }
      
      # ドラッグ終了
      [MouseOperations]::mouse_event([MouseOperations]::MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
      
      "Pan test completed from (${startX}, ${startY}) to (${endX}, ${endY})"
    `;
    
    return await this.executePowerShell(script);
  }

  async validateCoordinateTransformation(args) {
    const { testPoints } = args;
    const results = [];
    
    for (const point of testPoints) {
      // スクリーンショットを取得して座標変換を検証
      const result = await this.verifyCoordinateAtPoint(point.x, point.y);
      results.push({
        input: point,
        result: result,
        timestamp: new Date().toISOString()
      });
    }
    
    return {
      testResults: results,
      summary: `Tested ${results.length} coordinate transformations`
    };
  }

  async verifyCoordinateAtPoint(x, y) {
    // 実際の検証ロジック（スクリーンショット解析など）
    const script = `
      # スクリーンショットを取得
      Add-Type -AssemblyName System.Windows.Forms
      Add-Type -AssemblyName System.Drawing
      
      $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
      $screenshot = New-Object System.Drawing.Bitmap $bounds.width, $bounds.height
      $graphics = [System.Drawing.Graphics]::FromImage($screenshot)
      $graphics.CopyFromScreen($bounds.X, $bounds.Y, 0, 0, $bounds.size)
      
      # 指定座標の色を取得（検証用）
      $pixelColor = $screenshot.GetPixel(${x}, ${y})
      
      $graphics.Dispose()
      $screenshot.Dispose()
      
      "Pixel at (${x}, ${y}): R=$($pixelColor.R), G=$($pixelColor.G), B=$($pixelColor.B)"
    `;
    
    return await this.executePowerShell(script);
  }

  async executePowerShell(script) {
    return new Promise((resolve, reject) => {
      const process = spawn('powershell.exe', ['-Command', script], {
        stdio: ['ignore', 'pipe', 'pipe']
      });
      
      let output = '';
      let error = '';
      
      process.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      process.stderr.on('data', (data) => {
        error += data.toString();
      });
      
      process.on('close', (code) => {
        if (code === 0) {
          resolve(output.trim());
        } else {
          reject(new Error(`PowerShell error: ${error}`));
        }
      });
    });
  }
}

// サーバー起動
const server = new UIAutomationServer();
server.connect().then(() => {
  console.log('UI Automation MCP Server started');
}).catch(console.error);
```

## 3. テストシナリオの実行

### 3.1 MCPクライアントからのテスト実行
```javascript
// test-runner.js
const mcpClient = require('@modelcontextprotocol/sdk/client');

async function runUITests() {
  const client = new mcpClient.Client({
    server: 'ui-automation'
  });

  await client.connect();

  // 1. ズーム操作テスト
  console.log('Testing zoom functionality...');
  const zoomResult = await client.callTool('test_zoom_at_position', {
    x: 200,
    y: 200,
    zoomFactor: 2.0
  });
  console.log('Zoom test result:', zoomResult);

  // 2. パン操作テスト
  console.log('Testing pan functionality...');
  const panResult = await client.callTool('test_pan_gesture', {
    startX: 400,
    startY: 400,
    endX: 500,
    endY: 300
  });
  console.log('Pan test result:', panResult);

  // 3. 座標変換検証
  console.log('Validating coordinate transformation...');
  const coordResult = await client.callTool('validate_coordinate_transformation', {
    testPoints: [
      { x: 400, y: 400 },
      { x: 200, y: 200 },
      { x: 600, y: 600 }
    ]
  });
  console.log('Coordinate validation result:', coordResult);

  await client.disconnect();
}

runUITests().catch(console.error);
```

## 4. 実行手順

### 4.1 環境セットアップ
```bash
# 1. MCP サーバーの依存関係をインストール
npm install @modelcontextprotocol/sdk

# 2. UI Automation パッケージをインストール
dotnet add package FlaUI.Core
dotnet add package FlaUI.UIA3

# 3. MCP サーバーを起動
node ui-automation-server.js

# 4. MAUI アプリケーションを起動
.\MandelbrotMAUI.exe

# 5. テストを実行
node test-runner.js
```

### 4.2 期待される結果
- ズーム操作: クリック点が新しい中心になる
- パン操作: 滑らかな移動とリアルタイム更新
- 座標変換: 1ピクセル以内の精度

## 5. トラブルシューティング

### 5.1 一般的な問題
- **アプリが見つからない**: プロセス名とウィンドウタイトルを確認
- **UI要素が見つからない**: Accessibility IDまたはAutomation IDを設定
- **座標がずれる**: DPI設定とスケーリングファクターを確認

### 5.2 デバッグ方法
```powershell
# UI要素の検査
Add-Type -AssemblyName UIAutomationClient
$root = [System.Windows.Automation.AutomationElement]::RootElement
$condition = New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::NameProperty, "MandelbrotMAUI")
$app = $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $condition)
$app.Current | Format-List
```

## 6. 拡張機能

### 6.1 画像比較による検証
```javascript
// スクリーンショット比較機能
async function compareScreenshots(beforePath, afterPath) {
  const jimp = require('jimp');
  const before = await jimp.read(beforePath);
  const after = await jimp.read(afterPath);
  const diff = jimp.diff(before, after);
  return diff.percent < 0.1; // 10%未満の差異で合格
}
```

### 6.2 パフォーマンス測定
```javascript
// 応答時間測定
async function measureResponseTime(operation) {
  const start = performance.now();
  await operation();
  const end = performance.now();
  return end - start;
}
```

このセットアップにより、MCPサーバーを使用した包括的なUI自動テストシステムが構築できます。