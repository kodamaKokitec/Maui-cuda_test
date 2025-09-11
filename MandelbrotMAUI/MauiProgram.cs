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
#endif

		return builder.Build();
	}
}