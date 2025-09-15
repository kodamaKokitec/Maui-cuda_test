using Microsoft.Extensions.Logging;
using MandelbrotMAUI.Models;
using MandelbrotMAUI.Services;

namespace MandelbrotMAUI;

public static class MauiProgram
{
	public static MauiApp CreateMauiApp()
	{
		var builder = MauiApp.CreateBuilder();
		builder
			.UseMauiApp<App>()
			.ConfigureFonts(fonts =>
			{
				fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
				fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
			});

		// Register services
		builder.Services.AddSingleton<IMandelbrotService, CudaMandelbrotService>();

#if DEBUG
		builder.Services.AddLogging();
		builder.Logging.AddDebug();
		
		// Enable CUDA debug console for Visual Studio debugging
		CudaDebugHelper.EnableConsoleOutput();
		CudaDebugHelper.Log("MAUI Application starting with CUDA debugging enabled");
#endif

		return builder.Build();
	}
}
