[CmdletBinding()]
param(
    [ValidateRange(0.25, 100.0)]
    [double]$MinFreeGB = 0.75
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "storage.ps1")
$storage = Initialize-SifaciStorageEnvironment

$logRoot = $storage.Logs
$pidPath = Join-Path $logRoot "titck-sync.pid"
$stdoutPath = Join-Path $logRoot "titck-sync.stdout.log"
$stderrPath = Join-Path $logRoot "titck-sync.stderr.log"
if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    if (Get-Process -Id $existingPid -ErrorAction SilentlyContinue) {
        Write-Host "TITCK senkronizasyonu zaten calisiyor. PID: $existingPid" -ForegroundColor Cyan
        exit 0
    }
}

$pythonCommand = $storage.VenvPython
if (-not (Test-Path -LiteralPath $pythonCommand)) {
    throw "D:\SifaciAI\.venv bulunamadi. Once scripts\setup_d_drive.ps1 betigini calistirin."
}

$culture = [System.Globalization.CultureInfo]::InvariantCulture
$minimumText = $MinFreeGB.ToString($culture)
$process = Start-Process `
    -FilePath $pythonCommand `
    -ArgumentList "scripts\sync_titck.py", "--resume", "--min-free-gb", $minimumText `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
Write-Host "TITCK senkronizasyonu arka planda baslatildi." -ForegroundColor Green
Write-Host "PID: $($process.Id)"
Write-Host "Veri: $($storage.Data)"
Write-Host "Log: $stdoutPath"
