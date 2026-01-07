# ============================================================================
# Pipeline 2a Runner for Windows (PowerShell)
# URL list via API interception (network-only)
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TikTok Pipeline 2a - Windows" -ForegroundColor Cyan
Write-Host "URL List via API Interception" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# Account list (file in crawl_account/ folder, without .txt)
$ACCOUNT_LIST = "list1"

# OR use direct path to account file:
# $ACCOUNT_FILE = "crawl_account/custom_list.txt"

# Lookback period (days)
$LOOKBACK_DAYS = 30

# Limit number of profiles (leave empty for all)
$MAX_PROFILES = $null

# Browser restart frequency (restart every N profiles)
$RESTART_EVERY = 5

# Headless mode (set to $true to hide browser)
$HEADLESS = $false

# Verbose logging (set to $true for detailed logs)
$VERBOSE = $true

# ============================================================================
# BUILD COMMAND
# ============================================================================

$args = @(
    "run_pipeline_2a.py"
)

# Account list
if ($ACCOUNT_FILE) {
    $args += "--account-file", $ACCOUNT_FILE
} else {
    $args += "--list", $ACCOUNT_LIST
}

# Settings
$args += "--lookback", $LOOKBACK_DAYS

if ($MAX_PROFILES) {
    $args += "--max-profiles", $MAX_PROFILES
}

$args += "--restart-every", $RESTART_EVERY

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
Write-Host "  Account list: $ACCOUNT_LIST"
Write-Host "  Lookback: $LOOKBACK_DAYS days"
if ($MAX_PROFILES) {
    Write-Host "  Max profiles: $MAX_PROFILES"
}
Write-Host "  Restart every: $RESTART_EVERY profiles"
Write-Host "  Headless: $HEADLESS"
Write-Host "  Verbose: $VERBOSE"
Write-Host ""

Write-Host "Command:" -ForegroundColor Yellow
Write-Host "  python $($args -join ' ')"
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Pipeline 2a..." -ForegroundColor Green
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
    Write-Host "Pipeline 2a Completed" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Duration: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s"
    Write-Host "Output saved to: video_list\"
    Write-Host ""
    
} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Error Running Pipeline 2a" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
}

# Keep window open
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")