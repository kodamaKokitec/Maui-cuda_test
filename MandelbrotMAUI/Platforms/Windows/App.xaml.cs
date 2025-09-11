using Microsoft.UI.Xaml;
using System.Runtime.InteropServices;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace MandelbrotMAUI.WinUI;

/// <summary>
/// Provides application-specific behavior to supplement the default Application class.
/// </summary>
public partial class App : MauiWinUIApplication
{
	[DllImport("kernel32.dll", SetLastError = true)]
	[return: MarshalAs(UnmanagedType.Bool)]
	static extern bool AllocConsole();

	/// <summary>
	/// Initializes the singleton application object.  This is the first line of authored code
	/// executed, and as such is the logical equivalent of main() or WinMain().
	/// </summary>
	public App()
	{
#if DEBUG
		// Allocate a console for debug output
		AllocConsole();
		Console.WriteLine("=== Debug Console Allocated ===");
#endif
		this.InitializeComponent();
		Console.WriteLine("=== Windows App Initialized ===");
	}

	protected override MauiApp CreateMauiApp() => MauiProgram.CreateMauiApp();
}

