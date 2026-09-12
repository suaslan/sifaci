[CmdletBinding()]
param(
    [switch]$SkipDependencies
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "storage.ps1")
$storage = Initialize-SifaciStorageEnvironment

if (-not (Test-Path -LiteralPath $storage.VenvPython)) {
    $pyLauncher = (Get-Command py.exe -ErrorAction Stop).Source
    & $pyLauncher -3.12 -m venv $storage.Venv
    if ($LASTEXITCODE -ne 0) {
        throw "D: surucusundeki Python sanal ortami olusturulamadi."
    }
}

if (-not $SkipDependencies) {
    & $storage.VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip guncellenemedi." }
    & $storage.VenvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Python bagimliliklari kurulamadi." }
}

$seedMedicines = Join-Path $projectRoot "data\medicines"
if (Test-Path -LiteralPath $seedMedicines) {
    Copy-Item -Path (Join-Path $seedMedicines "*.json") -Destination $storage.Medicines -Force -ErrorAction SilentlyContinue
    Copy-Item -Path (Join-Path $seedMedicines "*.csv") -Destination $storage.Medicines -Force -ErrorAction SilentlyContinue
}

$seedExamples = Join-Path $projectRoot "data\examples"
if (Test-Path -LiteralPath $seedExamples) {
    $exampleTarget = Join-Path $storage.Data "examples"
    New-Item -ItemType Directory -Path $exampleTarget -Force | Out-Null
    Copy-Item -Path (Join-Path $seedExamples "*") -Destination $exampleTarget -Recurse -Force
}

Initialize-SifaciNextCacheJunction -FrontendRoot (Join-Path $projectRoot "frontend") -Target $storage.Next | Out-Null

Write-Host "Sifaci AI D: depolamasi hazir." -ForegroundColor Green
Write-Host "Kok: $($storage.Root)"
Write-Host "Python: $($storage.VenvPython)"
Write-Host "Veritabani: $(Join-Path $storage.Data 'medicines.db')"
Write-Host "Next.js cache: $($storage.Next)"
