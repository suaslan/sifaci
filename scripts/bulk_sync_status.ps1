[CmdletBinding()]
param(
    [ValidateRange(1, 200)]
    [int]$LogLines = 20
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "storage.ps1")
$storage = Initialize-SifaciStorageEnvironment
$logRoot = $storage.Logs
$pidPath = Join-Path $logRoot "titck-sync.pid"
$stdoutPath = Join-Path $logRoot "titck-sync.stdout.log"
$stderrPath = Join-Path $logRoot "titck-sync.stderr.log"

$running = $false
$syncPid = $null
if (Test-Path -LiteralPath $pidPath) {
    $syncPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $running = $null -ne (Get-Process -Id $syncPid -ErrorAction SilentlyContinue)
}
Write-Host "Calisiyor: $running"
if ($syncPid) {
    Write-Host "PID: $syncPid"
}

$pythonCommand = $storage.VenvPython
if (-not (Test-Path -LiteralPath $pythonCommand)) {
    throw "D:\SifaciAI\.venv bulunamadi. Once scripts\setup_d_drive.ps1 betigini calistirin."
}
& $pythonCommand scripts\pipeline_status.py

if (Test-Path -LiteralPath $stdoutPath) {
    Write-Host "`nSon cikti:"
    Get-Content -LiteralPath $stdoutPath -Tail $LogLines
}
if (Test-Path -LiteralPath $stderrPath) {
    $errors = Get-Content -LiteralPath $stderrPath -Tail $LogLines
    if ($errors) {
        Write-Host "`nSon hata/progress kaydi:"
        $errors
    }
}
