# Установка People Counter v2 как Windows-службы через NSSM (спека §8:
# нативный сервис для одиночной edge-машины проще и надёжнее Docker/WSL2).
# NSSM: https://nssm.cc/download → положите nssm.exe в PATH или рядом.
# Запуск из корня проекта в PowerShell от администратора:
#   .\deploy\install_service.ps1
param(
    [string]$ServiceName = "PeopleCounter",
    [string]$ConfigPath = "config\app.yaml"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Не найден venv: $python. Сначала: py -3.12 -m venv .venv; .venv\Scripts\pip install -e ."
}

$nssm = Get-Command nssm -ErrorAction SilentlyContinue
if (-not $nssm) { Write-Error "nssm.exe не найден в PATH (https://nssm.cc)" }

& nssm install $ServiceName $python "-m" "people_counter" "run" "--config" $ConfigPath
& nssm set $ServiceName AppDirectory $root
& nssm set $ServiceName DisplayName "People Counter v2"
& nssm set $ServiceName Description "Счёт вход/выход/occupancy по камерам (RF-DETR + BoT-SORT)"
& nssm set $ServiceName Start SERVICE_AUTO_START
# рестарт при падении: 24/7 (§8); внутренние компоненты супервизор чинит сам
& nssm set $ServiceName AppExit Default Restart
& nssm set $ServiceName AppRestartDelay 5000
# стоп по Ctrl+C (корректная финализация треков и дамп леджера)
& nssm set $ServiceName AppStopMethodConsole 15000
& nssm set $ServiceName AppStdout (Join-Path $root "logs\service_stdout.log")
& nssm set $ServiceName AppStderr (Join-Path $root "logs\service_stderr.log")
& nssm set $ServiceName AppRotateFiles 1
& nssm set $ServiceName AppRotateBytes 10485760

Write-Host "Служба '$ServiceName' установлена. Старт: nssm start $ServiceName"
Write-Host "Удаление: nssm remove $ServiceName confirm"
