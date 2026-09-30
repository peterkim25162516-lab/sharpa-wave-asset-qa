[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\external\sharpa-urdf-usd-xml')
)

$ErrorActionPreference = 'Stop'
$repository = 'https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git'
$commit = '6eea427eb24189519f32b9f21674cd534d3f973c'
$resolvedDestination = [System.IO.Path]::GetFullPath($Destination)

if (Test-Path -LiteralPath $resolvedDestination) {
    if (-not (Test-Path -LiteralPath (Join-Path $resolvedDestination '.git'))) {
        throw "Destination exists but is not a Git checkout: $resolvedDestination"
    }
} else {
    git -c core.longpaths=true clone --filter=blob:none --no-checkout $repository $resolvedDestination
    if ($LASTEXITCODE -ne 0) { throw "git clone failed with exit code $LASTEXITCODE" }
}

git -C $resolvedDestination config core.longpaths true
if ($LASTEXITCODE -ne 0) { throw "git config failed with exit code $LASTEXITCODE" }
git -C $resolvedDestination sparse-checkout init --cone
if ($LASTEXITCODE -ne 0) { throw "git sparse-checkout init failed with exit code $LASTEXITCODE" }
git -C $resolvedDestination sparse-checkout set `
    wave_01/left_sharpa_wave `
    wave_01/right_sharpa_wave `
    wave_01/dual_sharpa_wave `
    wave_01/sharpa_wave_float_base_urdf_usd
if ($LASTEXITCODE -ne 0) { throw "git sparse-checkout set failed with exit code $LASTEXITCODE" }
git -C $resolvedDestination fetch origin $commit --depth 1
if ($LASTEXITCODE -ne 0) { throw "git fetch failed with exit code $LASTEXITCODE" }
git -C $resolvedDestination checkout --detach $commit
if ($LASTEXITCODE -ne 0) { throw "git checkout failed with exit code $LASTEXITCODE" }

$actualCommit = git -C $resolvedDestination rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw "git rev-parse failed with exit code $LASTEXITCODE" }
if ($actualCommit -ne $commit) {
    throw "Pinned revision mismatch: expected $commit, got $actualCommit"
}

Write-Host "Sharpa assets ready at $resolvedDestination"
Write-Host "Pinned commit: $actualCommit"
