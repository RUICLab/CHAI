param(
    [string[]]$Apps = @(
        "ar-iclib", "sobel-iclib", "stringsearch-iclib",
        "fft-iclib", "lqi-iclib", "bc-iclib"
    ),
    [ValidateSet("operator_only", "operator_only_hw_prompt", "chai_msp430")]
    [string[]]$Variants = @("operator_only_hw_prompt", "chai_msp430"),
    [ValidateRange(6, 2147483647)]
    [int]$BoCalls = 8,
    [int]$Seed = 2026,
    [string]$RunId = "e2e_bo8_msp430",
    [string]$Model = "chai-qwen2.5-coder:7b-24k",
    [string]$OperatorCatalog = "checkmate_plus_10",
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Runs share transient target/, logs/, and output directories, so concurrent
# or orphaned campaign containers corrupt one another. Remove only containers
# created and labelled by this script before touching the workspace.
$orphanIds = @(& docker ps -aq --filter "label=chai.experiment=true" 2>$null)
foreach ($orphanId in $orphanIds) {
    if ($orphanId) {
        Write-Host "Stopping orphaned CHAI experiment container $orphanId"
        & docker rm -f $orphanId | Out-Null
    }
}

Write-Host "Running deterministic six-benchmark preflight..."
& docker run --rm `
    -e "LLM_MODEL=$Model" `
    -e "OLLAMA_MIN_CONTEXT=18000" `
    -v "${RepoRoot}:/workspace" `
    -w /workspace `
    chai-dev:local python3 scripts/preflight_six_benchmarks.py
if ($LASTEXITCODE -ne 0) {
    throw "Campaign preflight failed; no model-driven runs were started."
}
if ($PreflightOnly) {
    Write-Host "Preflight-only run completed; no model-driven runs were started."
    return
}

$ResultsRoot = Join-Path $RepoRoot "experiment_results/$RunId"
New-Item -ItemType Directory -Force $ResultsRoot | Out-Null

@{
    generated_at = (Get-Date).ToString("o")
    model = $Model
    ollama_min_context = 18000
    ollama_max_tokens = 2048
    ollama_max_retries = 0
    operator_catalog = $OperatorCatalog
    trace = "traces/RF_2.csv"
    capacitor = "68e-6"
    bo_calls = $BoCalls
    bo_seed = $Seed
    hardware = "config/hardware_msp430.json"
    variants = $Variants
    apps = $Apps
} | ConvertTo-Json | Set-Content (Join-Path $ResultsRoot "manifest.json")

function Copy-RunArtifacts([string]$App, [string]$Variant, [string]$Destination) {
    New-Item -ItemType Directory -Force "$Destination/logs" | Out-Null
    New-Item -ItemType Directory -Force "$Destination/outputs" | Out-Null

    $namePatterns = @(
        "$App*",
        "best_knobs_$App.csv",
        "experiment_metrics_$App*",
        "optimization_rejections_$App*",
        "generation_rejections_$App*",
        "generation_repairs_$App*",
        "target_rejections_$App*"
    )
    foreach ($pattern in $namePatterns) {
        Get-ChildItem logs -File -Filter $pattern -ErrorAction SilentlyContinue |
            Copy-Item -Destination "$Destination/logs" -Force
    }
    if (Test-Path "logs/original_checkpoints.txt") {
        Copy-Item "logs/original_checkpoints.txt" "$Destination/logs" -Force
    }

    Get-ChildItem outputs -File -Filter "*$App*" -ErrorAction SilentlyContinue |
        Copy-Item -Destination "$Destination/outputs" -Force
}

$status = @()
foreach ($App in $Apps) {
    foreach ($Variant in $Variants) {
        $Destination = Join-Path $ResultsRoot "$Variant/$App"
        New-Item -ItemType Directory -Force $Destination | Out-Null
        $ConsoleLog = Join-Path $Destination "console.log"
        $safeRunId = ($RunId -replace '[^A-Za-z0-9_.-]', '-')
        $safeApp = ($App -replace '[^A-Za-z0-9_.-]', '-')
        $containerName = "chai-$safeRunId-$Variant-$safeApp"
        @{
            app = $App
            variant = $Variant
            state = "running"
            started_at = (Get-Date).ToString("o")
        } | ConvertTo-Json | Set-Content (Join-Path $ResultsRoot "current.json")

        $dockerArgs = @(
            "run", "--rm", "--name", $containerName,
            "--label", "chai.experiment=true",
            "-e", "CHAI_BO_CALLS=$BoCalls",
            "-e", "CHAI_BO_SEED=$Seed",
            "-e", "LLM_MODEL=$Model",
            "-e", "OLLAMA_MIN_CONTEXT=18000",
            "-e", "OLLAMA_MAX_TOKENS=2048",
            "-e", "OLLAMA_MAX_RETRIES=0",
            "-v", "${RepoRoot}:/workspace",
            "-w", "/workspace",
            "chai-dev:local", "python3", "-u", "main.py",
            "--bm_name", $App,
            "--trace", "traces/RF_2.csv",
            "--trace_approx", "1",
            "--operator_catalog", $OperatorCatalog
        )
        if ($Variant -eq "chai_msp430") {
            $dockerArgs += @("--hardware", "config/hardware_msp430.json")
        } elseif ($Variant -eq "operator_only_hw_prompt") {
            $dockerArgs += @(
                "--hardware", "config/hardware_msp430.json",
                "--hardware_control", "prompt_only"
            )
        }

        Write-Host "[$(Get-Date -Format s)] $Variant / $App"
        try {
            & docker @dockerArgs 2>&1 | Tee-Object -FilePath $ConsoleLog | Out-Host
            $exitCode = $LASTEXITCODE
        } finally {
            # Normal --rm runs are already gone; this also stops the child if
            # PowerShell receives Ctrl+C or an exception interrupts the pipe.
            & docker rm -f $containerName 2>$null | Out-Null
        }
        try {
            Copy-RunArtifacts $App $Variant $Destination
        } catch {
            $_ | Out-File -Append $ConsoleLog
        }
        $status += [PSCustomObject]@{
            app = $App
            variant = $Variant
            exit_code = $exitCode
            state = $(if ($exitCode -eq 0) { "completed" } else { "failed" })
            completed_at = (Get-Date).ToString("o")
        }
        $status | ConvertTo-Json | Set-Content (Join-Path $ResultsRoot "status.json")
    }
}

@{
    state = "finished"
    completed_variants = $status.Count
    finished_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content (Join-Path $ResultsRoot "current.json")
