[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArguments
)

$ErrorActionPreference = "Stop"
$utf8Encoding = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8Encoding
[Console]::OutputEncoding = $utf8Encoding
$OutputEncoding = $utf8Encoding
$env:PYTHONIOENCODING = "utf-8"
. (Join-Path $PSScriptRoot "storage.ps1")
$storage = Initialize-SifaciStorageEnvironment
if (-not (Test-Path -LiteralPath $storage.VenvPython)) {
    throw "D:\SifaciAI\.venv bulunamadi. Once scripts\setup_d_drive.ps1 betigini calistirin."
}

& $storage.VenvPython @PythonArguments
exit $LASTEXITCODE
