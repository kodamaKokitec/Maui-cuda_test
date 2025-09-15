<<<<<<< HEAD
﻿using Microsoft.Extensions.Logging;
=======
>>>>>>> 714a192637bdc28463b85e4fc8f387b4f517cf83
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

<<<<<<< HEAD
		// DIは使用せず、App.xaml.csで直接インスタンス化する
		// builder.Services.AddSingleton<IMandelbrotService, CpuMandelbrotService>();
		// builder.Services.AddSingleton<TileManager>();

#if DEBUG
		builder.Logging.AddDebug();
		builder.Logging.SetMinimumLevel(LogLevel.Trace);
=======
		// Register services
		builder.Services.AddSingleton<IMandelbrotService, CudaMandelbrotService>();

#if DEBUG
		builder.Services.AddLogging();
		builder.Logging.AddDebug();
>>>>>>> 714a192637bdc28463b85e4fc8f387b4f517cf83
#endif

		return builder.Build();
	}
<<<<<<< HEAD
}
=======
}
>>>>>>> 714a192637bdc28463b85e4fc8f387b4f517cf83
