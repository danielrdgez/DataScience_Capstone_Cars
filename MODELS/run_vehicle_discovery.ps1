param(
    [string]$Python = '.\.venv-youtube\Scripts\python.exe',
    [string]$DataDir = 'DATA/CAR_DATA_FINAL',
    [string]$RunRoot = '',
    [string]$EmbeddingModel = '',
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto',
    [switch]$Resume,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $WorkspaceRoot
$Python = (Resolve-Path -LiteralPath $Python).Path
$DataDir = (Resolve-Path -LiteralPath $DataDir).Path
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    if ($Resume) { throw 'Supply the original -RunRoot when resuming.' }
    $RunRoot = Join-Path $WorkspaceRoot ('MODEL_OUTPUT/vehicle_similarity_v6/' + (Get-Date -Format 'yyyyMMddTHHmmss'))
}
if (-not [IO.Path]::IsPathRooted($RunRoot)) { $RunRoot = Join-Path $WorkspaceRoot $RunRoot }
$RunRoot = [IO.Path]::GetFullPath($RunRoot)
if ((Test-Path -LiteralPath $RunRoot) -and -not $Resume) {
    throw "Run directory already exists. Use -Resume with this path or choose a fresh directory: $RunRoot"
}

function Invoke-PythonStep {
    param([string]$Label, [string[]]$Arguments)
    Write-Host "`n$Label" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stopped after '$Label' (exit $LASTEXITCODE). No later stage ran." }
}

function Invoke-ArtifactStep {
    param([string]$Label, [string]$OutputPath, [string[]]$Arguments)
    if (Test-Path -LiteralPath (Join-Path $OutputPath 'manifest.json')) {
        if (-not $Resume) { throw "Completed stage already exists: $OutputPath" }
        Write-Host "Keeping completed stage: $Label"
        return
    }
    if (Test-Path -LiteralPath (Join-Path $OutputPath 'stage_state.json')) {
        if (-not $Resume) { throw "Interrupted stage exists; use -Resume: $OutputPath" }
        $Arguments += '--resume'
    }
    Invoke-PythonStep -Label $Label -Arguments $Arguments
}

Write-Host "Run root: $RunRoot" -ForegroundColor Green
if (-not $SkipTests) {
    Invoke-PythonStep -Label '0. Check regression dependencies' -Arguments @('-m', 'MODELS.vehicle_similarity_v6', 'check-dependencies')
}
Invoke-PythonStep -Label '1. Read-only source preflight' -Arguments @('-m', 'MODELS.vehicle_similarity_v6', 'preflight', '--data-dir', $DataDir)

if (-not $SkipTests) {
    Invoke-PythonStep -Label '2. Syntax checks' -Arguments @('-m', 'compileall', '-q',
        'MODELS/discovery_artifacts.py', 'MODELS/vehicle_similarity_v6.py', 'NHTSA/discovery_text.py', 'NHTSA/text_features.py',
        'car_app/discovery.py', 'car_app/recommendation.py', 'car_app/requirements.py', 'car_app/views.py',
        'app_pages/dashboard.py', 'app_pages/recommendation.py')
    Invoke-PythonStep -Label '3. Focused regression suites' -Arguments @('-m', 'unittest',
        'MODELS.test_vehicle_profiles', 'MODELS.test_price_evaluation', 'car_app.test_dashboard')
}

$Prepared = Join-Path $RunRoot 'prepared'
$TextSmoke = Join-Path $RunRoot 'text-smoke'
$SpecsSmoke = Join-Path $RunRoot 'specs-smoke'
$BundleSmoke = Join-Path $RunRoot 'bundle-smoke'
$ValidationSmoke = Join-Path $RunRoot 'validation-smoke'
$TextFull = Join-Path $RunRoot 'text'
$SpecsFull = Join-Path $RunRoot 'specs'
$BundleFull = Join-Path $RunRoot 'bundle'
$Comparison = Join-Path $RunRoot 'peer-comparison'
$ValidationFull = Join-Path $RunRoot 'validation'
$LatestCutoff = Get-Date -Format 'yyyy-MM-dd'
foreach ($TextDirectory in @($TextSmoke, $TextFull)) {
    $StateFile = Join-Path $TextDirectory 'stage_state.json'
    if (Test-Path -LiteralPath $StateFile) {
        $State = Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json
        $LatestCutoff = [string]$State.configuration.latest_cutoff
        break
    }
}
$TextArguments = @('-m', 'NHTSA.text_features', 'vectors', '--prepared', $Prepared, '--data-dir', $DataDir,
    '--period', 'both', '--latest-cutoff', $LatestCutoff, '--device', $Device)
if (-not [string]::IsNullOrWhiteSpace($EmbeddingModel)) { $TextArguments += @('--embedding-model', $EmbeddingModel) }

Invoke-ArtifactStep -Label '4. VIN-linked model and cohort preparation' -OutputPath $Prepared -Arguments @(
    '-m', 'MODELS.vehicle_similarity_v6', 'prepare', '--data-dir', $DataDir, '--output', $Prepared)
Invoke-ArtifactStep -Label '5. Smoke text vectors for both periods' -OutputPath $TextSmoke -Arguments ($TextArguments + @('--output', $TextSmoke, '--smoke'))
Invoke-ArtifactStep -Label '6. Smoke specification fits' -OutputPath $SpecsSmoke -Arguments @(
    '-m', 'MODELS.vehicle_similarity_v6', 'fit-specs', '--prepared', $Prepared, '--output', $SpecsSmoke, '--smoke')
Invoke-ArtifactStep -Label '7. Smoke evidence bundle' -OutputPath $BundleSmoke -Arguments @(
    '-m', 'MODELS.vehicle_similarity_v6', 'attach-evidence', '--specs', $SpecsSmoke, '--text', $TextSmoke, '--output', $BundleSmoke)
Invoke-ArtifactStep -Label '8. Smoke validation' -OutputPath $ValidationSmoke -Arguments @(
    '-m', 'MODELS.vehicle_similarity_v6', 'validate', '--run', $BundleSmoke, '--output', $ValidationSmoke)
Invoke-ArtifactStep -Label '9. Full TF-IDF and embeddings' -OutputPath $TextFull -Arguments ($TextArguments + @('--output', $TextFull))
Invoke-ArtifactStep -Label '10. Full specification fits' -OutputPath $SpecsFull -Arguments @(
    '-m', 'MODELS.vehicle_similarity_v6', 'fit-specs', '--prepared', $Prepared, '--output', $SpecsFull)
Invoke-ArtifactStep -Label '11. Full evidence bundle and safety projections' -OutputPath $BundleFull -Arguments @(
    '-m', 'MODELS.vehicle_similarity_v6', 'attach-evidence', '--specs', $SpecsFull, '--text', $TextFull, '--output', $BundleFull)
Invoke-ArtifactStep -Label '12. Experimental peer comparison for review' -OutputPath $Comparison -Arguments @(
    '-m', 'MODELS.vehicle_similarity_v6', 'compare-peers', '--run', $BundleFull, '--output', $Comparison)
Invoke-ArtifactStep -Label '13. Full validation' -OutputPath $ValidationFull -Arguments @(
    '-m', 'MODELS.vehicle_similarity_v6', 'validate', '--run', $BundleFull, '--output', $ValidationFull)
Invoke-PythonStep -Label '14. Activate the full validated discovery bundle' -Arguments @(
    '-m', 'MODELS.vehicle_similarity_v6', 'activate', '--run', $BundleFull, '--validation', $ValidationFull)
Write-Host "`nCompleted and activated: $BundleFull" -ForegroundColor Green
Write-Host "Validation receipt: $(Join-Path $ValidationFull 'validation.json')"
Write-Host 'The Streamlit app has not been launched. Start it separately after reviewing this output.'
