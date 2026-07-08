# build_submission.ps1 — one-command submission pipeline.
# All logic lives in Python (rl/export.py, rl/gate.py); this only orchestrates.
#
# Build neural:      .\build_submission.ps1
# Build rule agent:  .\build_submission.ps1 -Agent rules
# Build + submit:    .\build_submission.ps1 -Agent rules -Message "M6.0 generic pilot + Lucario deck"
# (submitting is opt-in because Kaggle limits submissions per day)
param(
    [string]$Message,
    [ValidateSet("neural", "rules")]
    [string]$Agent = "neural"
)
$ErrorActionPreference = "Stop"
$Competition = "pokemon-tcg-ai-battle"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"

# Per-agent bundle layout.
if ($Agent -eq "rules") {
    $subDir = "submission_rules"
    $tarFiles = @("main.py", "deck.csv", "cg", "rl")
    $required = @("main.py", "deck.csv", "cg/api.py", "cg/libcg.so",
                  "rl/__init__.py", "rl/combat.py", "rl/generic_pilot.py")
} else {
    $subDir = "submission"
    $tarFiles = @("main.py", "deck.csv", "policy_weights.npz", "card_features.npy", "cg")
    $required = @("main.py", "deck.csv", "policy_weights.npz", "card_features.npy",
                  "cg/api.py", "cg/utils.py", "cg/sim.py", "cg/libcg.so")
}

Write-Host "[1/3] Export artifacts (rl.export --agent $Agent)" -ForegroundColor Cyan
& $py -m rl.export --agent $Agent
if ($LASTEXITCODE -ne 0) { throw "export failed" }

Write-Host "[2/3] Gates (rl.gate --agent $Agent)" -ForegroundColor Cyan
& $py -m rl.gate --agent $Agent
if ($LASTEXITCODE -ne 0) { throw "gates failed" }

Write-Host "[3/3] Package" -ForegroundColor Cyan
New-Item -ItemType Directory -Force dist | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = "dist\submission_${Agent}_$stamp.tar.gz"
tar -czf $out -C $subDir @tarFiles
if ($LASTEXITCODE -ne 0) { throw "tar failed" }

# The local gate can't catch a file missing from the BUNDLE (imports resolve from
# the project root here) -- so verify the archive contents explicitly.
$contents = tar -tzf $out
foreach ($req in $required) {
    if ($contents -notcontains $req) { throw "package is missing $req" }
}

$kb = [math]::Round((Get-Item $out).Length / 1KB)
Write-Host "`nBUILD OK -> $out ($kb KB)" -ForegroundColor Green

if (-not $Message) {
    Write-Host "No -Message given: skipping Kaggle upload. Submit later with:" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\kaggle.exe competitions submit -c $Competition -f $out -m `"<message>`"" -ForegroundColor Yellow
    exit 0
}

Write-Host "[4/4] Submitting to Kaggle" -ForegroundColor Cyan
$kaggle = Join-Path $PSScriptRoot ".venv\Scripts\kaggle.exe"
$hasToken = $env:KAGGLE_API_TOKEN -or
            (Test-Path "$env:USERPROFILE\.kaggle\kaggle.json") -or
            ($env:KAGGLE_USERNAME -and $env:KAGGLE_KEY)
if (-not $hasToken) {
    throw "No Kaggle credentials. Set KAGGLE_API_TOKEN (or KAGGLE_USERNAME/KAGGLE_KEY), or save an API token (kaggle.com -> Settings -> API) as $env:USERPROFILE\.kaggle\kaggle.json"
}
& $kaggle competitions submit -c $Competition -f $out -m $Message
if ($LASTEXITCODE -ne 0) { throw "kaggle submit failed" }

Write-Host "`nSUBMITTED. Recent submissions:" -ForegroundColor Green
& $kaggle competitions submissions -c $Competition -v | Select-Object -First 4
