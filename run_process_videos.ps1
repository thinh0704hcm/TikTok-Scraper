# ============================================================================
# Pipeline 2b Runner for Windows (PowerShell)
# Process URL list → metadata + comments (network-only)
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TikTok Pipeline 2b - Windows" -ForegroundColor Cyan
Write-Host "Process Videos: Metadata + Comments" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# URL list file (output from Pipeline 2a)
# Example: "video_list/20260107/phuongmychiofficial.txt"
$URL_FILE = "video_list\20260108_131623\all_videos.txt"

# Max comments per video
$MAX_COMMENTS = 100

# Browser restart frequency (restart every N videos)
$RESTART_EVERY = 20

# Memory-based restart (RSS MB threshold). Leave $null to disable.
$MEM_RESTART_MB = 2000

# Headless mode (set to $true to hide browser)
$HEADLESS = $false

# Verbose logging (set to $true for detailed logs)
$VERBOSE = $true

# ============================================================================
# BUILD COMMAND
# ============================================================================

$args = @(
    "run_pipeline_2b.py",
    "--file", $URL_FILE
)

# Settings
$args += "--max-comments", $MAX_COMMENTS
$args += "--restart-every", $RESTART_EVERY

if ($MEM_RESTART_MB) {
    $args += "--mem-restart-mb", $MEM_RESTART_MB
}

if ($HEADLESS) {
    $args += "--headless"
}

if ($VERBOSE) {
    $args += "--verbose"
}

# ============================================================================
# DISPLAY CONFIG
# ============================================================================

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  URL file: $URL_FILE"
Write-Host "  Max comments/video: $MAX_COMMENTS"
Write-Host "  Restart every: $RESTART_EVERY videos"
if ($MEM_RESTART_MB) {
    Write-Host "  Mem restart: $MEM_RESTART_MB MB"
}
Write-Host "  Headless: $HEADLESS"
Write-Host "  Verbose: $VERBOSE"
Write-Host ""

# Check if file exists
if (-not (Test-Path $URL_FILE)) {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Error: URL file not found!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "File: $URL_FILE" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run Pipeline 2a first to generate URL lists:" -ForegroundColor Yellow
    Write-Host "  pwsh ./run_comments_scraper.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host "Command:" -ForegroundColor Yellow
Write-Host "  python $($args -join ' ')"
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Pipeline 2b..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# RUN
# ============================================================================

try {
    $startTime = Get-Date
    
    & python $args
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Pipeline 2b Completed" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Duration: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s"
    Write-Host "Output saved to: comments_data\"
    Write-Host ""
    
} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Error Running Pipeline 2b" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
}

# Keep window open
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
