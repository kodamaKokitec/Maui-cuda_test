namespace MandelbrotMAUI;

public partial class AppShell : Shell
{
	public AppShell()
	{
		InitializeComponent();
		
		// MainPage�̃��[�g��o�^
		Routing.RegisterRoute("MainPage", typeof(MainPage));
		
		// �����y�[�W�ɒ��ڃi�r�Q�[�g
		CurrentItem = this.Items.First();
	}
}
