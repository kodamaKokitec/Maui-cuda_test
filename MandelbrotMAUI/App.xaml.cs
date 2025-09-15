using MandelbrotMAUI.Services;

namespace MandelbrotMAUI;

public partial class App : Application
{
	private void LogToFile(string message)
	{
		try
		{
			var logFile = Path.Combine(AppContext.BaseDirectory, "app_debug.log");
			File.AppendAllText(logFile, $"{DateTime.Now:HH:mm:ss.fff} - {message}\n");
		}
		catch
		{
			// ���O�t�@�C���������݂Ɏ��s���Ă��A�v�����N���b�V�������Ȃ�
		}
	}

	public App()
	{
		LogToFile("=== App Constructor Starting ===");
		Console.WriteLine("=== App Constructor Starting ===");
		InitializeComponent();
		LogToFile("=== App Constructor Completed ===");
		Console.WriteLine("=== App Constructor Completed ===");
	}

	protected override Window CreateWindow(IActivationState? activationState)
	{
		LogToFile("=== CreateWindow Called ===");
		Console.WriteLine("=== CreateWindow Called ===");
		
		try
		{
			LogToFile("Creating services...");
			// CUDA�T�[�r�X���g�p�iCUDA�����p�ł��Ȃ��ꍇ�͎����I��CPU�Ƀt�H�[���o�b�N�j
			var mandelbrotService = new CudaMandelbrotService();
			
			LogToFile($"Mandelbrot Service: {mandelbrotService.GetEngineInfo()}");
			LogToFile("Creating MainPage_Image...");
			var mainPage = new MainPage_Image();
			
			LogToFile("Creating Window...");
			var window = new Window(mainPage);
			
			// Window�C�x���g�̊Ď�
			window.Created += (sender, e) => LogToFile("Window.Created event fired");
			window.Activated += (sender, e) => LogToFile("Window.Activated event fired");
			window.Deactivated += (sender, e) => LogToFile("Window.Deactivated event fired");
			window.Stopped += (sender, e) => LogToFile("Window.Stopped event fired");
			window.Destroying += (sender, e) => LogToFile("Window.Destroying event fired");
			
			LogToFile("=== Window Created Successfully ===");
			Console.WriteLine("=== Window Created Successfully ===");
			return window;
		}
		catch (Exception ex)
		{
			LogToFile($"=== ERROR in CreateWindow: {ex.Message} ===");
			LogToFile($"Stack trace: {ex.StackTrace}");
			Console.WriteLine($"=== ERROR in CreateWindow: {ex.Message} ===");
			Console.WriteLine($"Stack trace: {ex.StackTrace}");
			
			// �t�H�[���o�b�N: �V���v���ȃy�[�W���쐬
			var fallbackPage = new ContentPage
			{
				Title = "Error",
				Content = new Label
				{
					Text = $"Error: {ex.Message}",
					HorizontalOptions = LayoutOptions.Center,
					VerticalOptions = LayoutOptions.Center
				}
			};
			return new Window(fallbackPage);
		}
	}
}
