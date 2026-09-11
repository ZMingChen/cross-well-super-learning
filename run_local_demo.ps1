$ErrorActionPreference = 'Stop'
$demoPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $demoPython)) {
    throw 'Create .venv and install requirements.txt as described in README.md first.'
}
& $demoPython (Join-Path $PSScriptRoot 'scripts\release_preflight.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
