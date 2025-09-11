using Microsoft.Extensions.Logging;
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

		// DIは使用せず、App.xaml.csで直接インスタンス化する
		// builder.Services.AddSingleton<IMandelbrotService, CpuMandelbrotService>();
		// builder.Services.AddSingleton<TileManager>();

#if DEBUG
		builder.Logging.AddDebug();
		builder.Logging.SetMinimumLevel(LogLevel.Trace);
#endif

		return builder.Build();
	}
}
