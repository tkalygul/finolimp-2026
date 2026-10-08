param(
    [Parameter(Mandatory=$true)][string]$Data,
    [string]$Out = "out/report",
    [switch]$SkipML
)
$ErrorActionPreference = "Stop"
$projectPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $projectPython)) {
    Write-Error "Создайте окружение из README: py -3.12 -m venv .venv; .venv\Scripts\python.exe -m pip install -r requirements-lock.txt"
    exit 1
}
$projectArguments = @((Join-Path $PSScriptRoot "reconcile.py"), "--data", $Data, "--out", $Out)
if ($SkipML) { $projectArguments += "--skip-ml" }
& $projectPython -X utf8 @projectArguments
exit $LASTEXITCODE
