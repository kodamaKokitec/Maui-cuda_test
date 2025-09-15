using MandelbrotMAUI.Services;
using MandelbrotMAUI.Views;
using MandelbrotMAUI.Models;

namespace MandelbrotMAUI;

public partial class MainPage : ContentPage
{
    private readonly TileManager _tileManager;
    private readonly MandelbrotCanvas _canvas;
    private readonly MandelbrotParameters _parameters;
    private readonly IMandelbrotService _mandelbrotService;

    private void LogToFile(string message)
    {
        try
        {
            var logFile = Path.Combine(AppContext.BaseDirectory, "mainpage_debug.log");
            File.AppendAllText(logFile, $"{DateTime.Now:HH:mm:ss.fff} - {message}\n");
        }
        catch
        {
            // ログファイル書き込みに失敗してもアプリをクラチE��ュさせなぁE
        }
    }

    public MainPage(IMandelbrotService mandelbrotService, TileManager tileManager)
    {
        try
        {
            LogToFile("=== MainPage Constructor Starting ===");
            Console.WriteLine("=== MainPage Constructor Starting ===");
            
            LogToFile("Calling InitializeComponent...");
            InitializeComponent();
            LogToFile("InitializeComponent completed");
            Console.WriteLine("InitializeComponent completed");
            
            _mandelbrotService = mandelbrotService;
            _tileManager = tileManager;
            LogToFile("Services assigned");
            Console.WriteLine("Services assigned");
            
            // パラメータの初期匁E
            _parameters = new MandelbrotParameters();
            BindingContext = _parameters;
            LogToFile("Parameters and BindingContext set");
            Console.WriteLine("Parameters and BindingContext set");
            
            LogToFile("Creating MandelbrotCanvas...");
            // キャンバスの作�Eと設宁E
            _canvas = new MandelbrotCanvas(_tileManager);
            LogToFile("Canvas created");
            Console.WriteLine("Canvas created");
            
            LogToFile("Assigning canvas to container...");
            CanvasContainer.Content = _canvas;
            LogToFile("Canvas assigned to container");
            Console.WriteLine("Canvas assigned to container");
            
            LogToFile("Setting engine info...");
            // エンジン惁E��の表示
            EngineLabel.Text = _mandelbrotService.GetEngineInfo();
            LogToFile($"Engine info set: {_mandelbrotService.GetEngineInfo()}");
            Console.WriteLine($"Engine info set: {_mandelbrotService.GetEngineInfo()}");
            
            LogToFile("Setting up parameter change monitoring...");
            // パラメータ変更の監要E
            _parameters.PropertyChanged += OnParametersChanged;
            LogToFile("=== MainPage Constructor Completed Successfully ===");
            Console.WriteLine("=== MainPage Constructor Completed Successfully ===");
        }
        catch (Exception ex)
        {
            LogToFile($"=== ERROR in MainPage Constructor: {ex.Message} ===");
            LogToFile($"Stack trace: {ex.StackTrace}");
            Console.WriteLine($"=== ERROR in MainPage Constructor: {ex.Message} ===");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
            throw;
        }
    }

    private void OnParametersChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        // パラメータが変更されたら再描画
        _canvas.Invalidate();
    }

    private void OnResetClicked(object? sender, EventArgs e)
    {
        _parameters.CenterX = -0.5;
        _parameters.CenterY = 0.0;
        _parameters.Zoom = 1.0;
        _parameters.MaxIterations = 1000;
        
        _tileManager.ClearCache();
        _canvas.Invalidate();
    }

    private void OnZoomInClicked(object? sender, EventArgs e)
    {
        _canvas.OnZoomGesture(2.0, Width / 2, Height / 2);
    }

    private void OnZoomOutClicked(object? sender, EventArgs e)
    {
        _canvas.OnZoomGesture(0.5, Width / 2, Height / 2);
    }
}
