param(
    [string]$Python = "C:\Users\saint\cqfenv\python.exe",
    [switch]$RefreshOfficial,
    [switch]$RefreshCrowd,
    [switch]$ExecuteNotebook
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

function Invoke-ProjectPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Push-Location -LiteralPath $projectRoot
try {
    if ($RefreshOfficial) {
        Invoke-ProjectPython "-m" "visa_wait.cli" "official-refresh"
    }
    if ($RefreshCrowd) {
        $snapshot = Get-Date -Format "yyyy-MM-dd"
        Invoke-ProjectPython "-m" "visa_wait.cli" "collect" "visadashboard" "--snapshot-date" $snapshot
        Invoke-ProjectPython "-m" "visa_wait.cli" "profile"
    }

    Invoke-ProjectPython "-m" "visa_wait.cli" "validate"
    Invoke-ProjectPython "-m" "visa_wait.cli" "audit"
    Invoke-ProjectPython "-m" "visa_wait.cli" "train"
    Invoke-ProjectPython "-m" "visa_wait.cli" "visualize"
    Invoke-ProjectPython "-m" "unittest" "discover" "-s" "tests" "-v"
    Invoke-ProjectPython "-m" "visa_wait.cli" "forecast" `
        "--lodged-at" "2026-09-24T15:30+08:00" `
        "--as-of" "2026-09-24T15:30+08:00" `
        "--education-level" "PhD" `
        "--study-sector" "Postgraduate Research" `
        "--submit-location" "Outside Australia"

    if ($ExecuteNotebook) {
        Invoke-ProjectPython "scripts/execute_notebook.py" `
            "notebooks/01_data_audit_and_survival_baselines.ipynb" `
            "--kernel" "cqfenv" "--timeout" "300" `
            "--html-output" "reports/notebook_data_audit_and_survival.html"
    }
}
finally {
    Pop-Location
}
