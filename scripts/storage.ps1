[CmdletBinding()]
param()

function Initialize-SifaciStorageEnvironment {
    [CmdletBinding()]
    param(
        [string]$StorageRoot = "D:\SifaciAI"
    )

    if (-not (Test-Path -LiteralPath "D:\")) {
        throw "D: surucusu bulunamadi. Sifaci AI calisma dosyalari D:\SifaciAI altinda olmalidir."
    }

    $root = [System.IO.Path]::GetFullPath($StorageRoot).TrimEnd('\')
    if (-not $root.Equals("D:\SifaciAI", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Sifaci AI depolama koku D:\SifaciAI olmalidir. Gecersiz yol: $root"
    }

    $paths = [ordered]@{
        Root                 = $root
        Data                 = Join-Path $root "data"
        Medicines            = Join-Path $root "data\medicines"
        Documents            = Join-Path $root "data\documents"
        Source               = Join-Path $root "data\source"
        Runtime              = Join-Path $root "runtime"
        Logs                 = Join-Path $root "logs"
        Temp                 = Join-Path $root "temp"
        Cache                = Join-Path $root "cache"
        HuggingFace          = Join-Path $root "cache\huggingface"
        HuggingFaceHub       = Join-Path $root "cache\huggingface\hub"
        Transformers         = Join-Path $root "cache\huggingface\transformers"
        SentenceTransformers = Join-Path $root "cache\huggingface\sentence-transformers"
        Torch                = Join-Path $root "cache\torch"
        Pip                  = Join-Path $root "cache\pip"
        Uv                   = Join-Path $root "cache\uv"
        Npm                  = Join-Path $root "cache\npm"
        TypeScript           = Join-Path $root "cache\typescript"
        Python               = Join-Path $root "cache\python"
        Pytest               = Join-Path $root "cache\pytest"
        Next                 = Join-Path $root "cache\nextjs"
        Foundry              = Join-Path $root "foundry"
        Models               = Join-Path $root "models"
        Playwright           = Join-Path $root "playwright-browsers"
        Venv                 = Join-Path $root ".venv"
        VenvPython           = Join-Path $root ".venv\Scripts\python.exe"
    }

    foreach ($path in $paths.Values) {
        if ($path -ne $paths.VenvPython -and $path -ne $paths.Next) {
            New-Item -ItemType Directory -Path $path -Force | Out-Null
        }
    }

    $env:SIFACI_STORAGE_ROOT = $paths.Root
    $env:SIFACI_DATA_DIR = $paths.Data
    $env:SIFACI_RUNTIME_DIR = $paths.Runtime
    $env:SIFACI_TEMP_DIR = $paths.Temp
    $env:SIFACI_CACHE_DIR = $paths.Cache
    $env:FOUNDRY_APP_DATA_DIR = $paths.Foundry
    $env:FOUNDRY_MODEL_CACHE_DIR = $paths.Models
    $env:FOUNDRY_LOGS_DIR = $paths.Logs
    $env:HF_HOME = $paths.HuggingFace
    $env:HF_HUB_CACHE = $paths.HuggingFaceHub
    $env:TRANSFORMERS_CACHE = $paths.Transformers
    $env:SENTENCE_TRANSFORMERS_HOME = $paths.SentenceTransformers
    $env:TORCH_HOME = $paths.Torch
    $env:XDG_CACHE_HOME = $paths.Cache
    $env:PIP_CACHE_DIR = $paths.Pip
    $env:UV_CACHE_DIR = $paths.Uv
    $env:PLAYWRIGHT_BROWSERS_PATH = $paths.Playwright
    $env:NPM_CONFIG_CACHE = $paths.Npm
    $env:npm_config_cache = $paths.Npm
    $env:PYTHONPYCACHEPREFIX = $paths.Python
    $env:NODE_PATH = Join-Path (Split-Path -Parent $PSScriptRoot) "frontend\node_modules"
    $env:TEMP = $paths.Temp
    $env:TMP = $paths.Temp
    $env:TMPDIR = $paths.Temp
    $env:PYTHONUNBUFFERED = "1"

    return [pscustomobject]$paths
}

function Initialize-SifaciNextCacheJunction {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$FrontendRoot,
        [Parameter(Mandatory)][string]$Target
    )

    $frontend = [System.IO.Path]::GetFullPath($FrontendRoot).TrimEnd('\')
    $localCache = Join-Path $frontend ".next"
    $targetCache = [System.IO.Path]::GetFullPath($Target).TrimEnd('\')
    if (-not $targetCache.StartsWith("D:\SifaciAI\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Next.js cache hedefi D:\SifaciAI altinda olmalidir: $targetCache"
    }

    if (Test-Path -LiteralPath $localCache) {
        $item = Get-Item -LiteralPath $localCache -Force
        if ($item.LinkType -eq "Junction") {
            $currentTarget = [System.IO.Path]::GetFullPath([string]$item.Target).TrimEnd('\')
            if ($currentTarget.Equals($targetCache, [System.StringComparison]::OrdinalIgnoreCase)) {
                return $localCache
            }
            throw "frontend\.next baska bir hedefe bagli: $currentTarget"
        }

        if (Test-Path -LiteralPath $targetCache) {
            $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
            $targetCache = Join-Path (Split-Path -Parent $targetCache) "nextjs-migrated-$stamp"
        }
        Move-Item -LiteralPath $localCache -Destination $targetCache
    } else {
        New-Item -ItemType Directory -Path $targetCache -Force | Out-Null
    }

    New-Item -ItemType Junction -Path $localCache -Target $targetCache | Out-Null
    return $localCache
}
