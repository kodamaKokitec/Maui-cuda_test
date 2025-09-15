namespace MandelbrotMAUI;

public partial class AppShell : Shell
{
	public AppShell()
	{
		InitializeComponent();
		
		// MainPageのルートを登録
		Routing.RegisterRoute("MainPage", typeof(MainPage));
		
		// 初期ページに直接ナビゲート
		CurrentItem = this.Items.First();
	}
}
