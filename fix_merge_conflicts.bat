@echo off
echo Removing Git merge conflict markers...

REM Remove merge conflict markers from all files
for %%f in (*.bat *.cu *.cs *.xaml *.csproj *.h *.def) do (
    if exist "%%f" (
        echo Processing %%f...
        powershell -Command "(Get-Content '%%f') | Where-Object { $_ -notmatch '^<<<<<<< HEAD$|^=======$|^>>>>>>> [0-9a-f]+$' } | Set-Content '%%f'"
    )
)

REM Process files in subdirectories
for /r . %%f in (*.bat *.cu *.cs *.xaml *.csproj *.h *.def) do (
    if exist "%%f" (
        echo Processing %%f...
        powershell -Command "(Get-Content '%%f') | Where-Object { $_ -notmatch '^<<<<<<< HEAD$|^=======$|^>>>>>>> [0-9a-f]+$' } | Set-Content '%%f'"
    )
)

echo Merge conflict markers removed!
pause
