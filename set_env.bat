@echo off
REM CUDA Toolkitのパスを追加
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\libnvvp;%PATH%

REM Visual StudioのMSBuildパスを追加
set PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.38.33130\bin;%PATH%

REM そ�E他忁E��なパスを追加�E�侁E プロジェクト固有�EチE�Eル�E�E
set PATH=%CD%;%PATH%

@echo 環墁E��数を設定しました�
