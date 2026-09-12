[CmdletBinding()]
param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$utf8Encoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8Encoding
$OutputEncoding = $utf8Encoding
$env:PYTHONIOENCODING = "utf-8"
$projectRoot = $PSScriptRoot
$frontendRoot = Join-Path $projectRoot "frontend"
. (Join-Path $projectRoot "scripts\storage.ps1")
$storage = Initialize-SifaciStorageEnvironment
$runtimeRoot = $storage.Runtime
$pythonCommand = $storage.VenvPython
if (-not (Test-Path -LiteralPath $pythonCommand)) {
    throw "D:\SifaciAI\.venv bulunamadi. Once scripts\setup_d_drive.ps1 betigini calistirin."
}
& $pythonCommand -c "import fastapi, uvicorn, foundry_local_sdk" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "D:\SifaciAI\.venv icindeki paketler eksik. scripts\setup_d_drive.ps1 betigini calistirin."
}
$npmCommand = (Get-Command npm.cmd -ErrorAction Stop).Source

Initialize-SifaciNextCacheJunction -FrontendRoot $frontendRoot -Target $storage.Next | Out-Null

$dataRoot = & $pythonCommand -c "from config import DATA_DIR; print(DATA_DIR)"
if ($LASTEXITCODE -ne 0) {
    throw "Veri klasoru belirlenemedi."
}
$dataDrive = [System.IO.DriveInfo]::new((Split-Path -Qualifier $dataRoot))
$freeDiskGB = [math]::Round($dataDrive.AvailableFreeSpace / 1GB, 2)
if ($freeDiskGB -lt 0.25) {
    throw "Veri diskindeki alan kritik duzeyde ($freeDiskGB GB). Sifaci AI baslatilmadan once yer acin."
}
if ($freeDiskGB -lt 2) {
    Write-Host "[UYARI] Veri diskinde yalnizca $freeDiskGB GB bos alan var." -ForegroundColor Yellow
}
Write-Host "[DEPOLAMA] $dataRoot ($freeDiskGB GB bos)"

function Test-SifaciEndpoint {
    param([Parameter(Mandatory)][string]$Uri)

    try {
        $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    } catch {
        return $false
    }
}

function Wait-SifaciEndpoint {
    param(
        [Parameter(Mandatory)][string]$Uri,
        [Parameter(Mandatory)][string]$ServiceName,
        [int]$TimeoutSeconds = 90
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-SifaciEndpoint -Uri $Uri) {
            Write-Host "[HAZIR] $ServiceName" -ForegroundColor Green
            return
        }
        Start-Sleep -Seconds 1
    }

    throw "$ServiceName baslatilamadi. Ayrinti icin D:\SifaciAI\runtime klasorundeki loglari kontrol edin."
}

$apiHealthUrl = "http://127.0.0.1:8000/health"
$frontendUrl = "http://localhost:3000"

Write-Host "[VERI] JSON/CSV ilac kayitlari kanonik veritabaniyla esitleniyor."
& $pythonCommand -m src.ingestion
if ($LASTEXITCODE -ne 0) {
    throw "Ilac ingestion islemi basarisiz oldu."
}

$databaseState = & $pythonCommand -c "import json; from config import DATABASE_PATH; from src.database import get_database_stats; print(json.dumps({'database': str(DATABASE_PATH), **get_database_stats()}, ensure_ascii=True))"
if ($LASTEXITCODE -ne 0) {
    throw "Veritabani durumu okunamadi."
}
$databaseInfo = $databaseState | ConvertFrom-Json
Write-Host "[VERITABANI] $($databaseInfo.database)"
Write-Host "[KAYIT] $($databaseInfo.medicine_count) ilac / $($databaseInfo.chunk_count) chunk"
if ($databaseInfo.medicine_count -le 0 -or $databaseInfo.chunk_count -le 0) {
    throw "Ilac veritabani bos. Dogrulanmis JSON dosyalarini D:\SifaciAI\data\medicines klasorune ekleyin."
}

if (Test-SifaciEndpoint -Uri $apiHealthUrl) {
    Write-Host "[AKTIF] Python API zaten calisiyor." -ForegroundColor Cyan
} else {
    Write-Host "[BASLATILIYOR] Python API"
    Start-Process `
        -FilePath $pythonCommand `
        -ArgumentList "api.py" `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $runtimeRoot "api.stdout.log") `
        -RedirectStandardError (Join-Path $runtimeRoot "api.stderr.log")
    Wait-SifaciEndpoint -Uri $apiHealthUrl -ServiceName "Python API"
}

if (Test-SifaciEndpoint -Uri $frontendUrl) {
    Write-Host "[AKTIF] Next.js frontend zaten calisiyor." -ForegroundColor Cyan
} else {
    Write-Host "[BASLATILIYOR] Next.js frontend"
    Start-Process `
        -FilePath $npmCommand `
        -ArgumentList "run", "dev" `
        -WorkingDirectory $frontendRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $runtimeRoot "frontend.stdout.log") `
        -RedirectStandardError (Join-Path $runtimeRoot "frontend.stderr.log")
    Wait-SifaciEndpoint -Uri $frontendUrl -ServiceName "Next.js frontend"
}

Write-Host ""
Write-Host "Sifaci AI hazir: $frontendUrl" -ForegroundColor Green
Write-Host "API saglik kontrolu: $apiHealthUrl"

if (-not $NoBrowser) {
    Start-Process $frontendUrl
}
