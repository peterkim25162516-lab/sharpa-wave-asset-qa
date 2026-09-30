[CmdletBinding()]
param(
    [string]$AssetRoot = (Join-Path $PSScriptRoot '..\external\sharpa-urdf-usd-xml\wave_01'),
    [string]$Output = (Join-Path $PSScriptRoot '..\results\v0.1')
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found. Follow the README setup first: $python"
}

& $python -m wave_asset_qa audit `
    ([System.IO.Path]::GetFullPath($AssetRoot)) `
    --upstream-commit 6eea427eb24189519f32b9f21674cd534d3f973c `
    --output ([System.IO.Path]::GetFullPath($Output))

exit $LASTEXITCODE
