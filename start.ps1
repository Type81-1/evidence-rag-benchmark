param(
    [string]$PythonPath = ".\.venv\Scripts\python.exe",
    [int]$Port = 8001
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python was not found at: $PythonPath. Create .venv first or pass -PythonPath with Python 3.11+."
}

& $PythonPath -m uvicorn app:app --host 127.0.0.1 --port $Port
