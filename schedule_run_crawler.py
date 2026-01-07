"""
Scheduled Pipeline 1 Scraper with Consistent Intervals

Runs Pipeline 1 (fast metadata scraping) at fixed intervals with proper 
error handling, logging, and consistent scheduling.
"""

import subprocess
import time
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Interval between runs (in seconds)
INTERVAL_SECONDS = 3600  # 1 hour
# INTERVAL_SECONDS = 1800  # 30 minutes
# INTERVAL_SECONDS = 7200  # 2 hours

# Pipeline 1 scraper settings
SCRAPER_SCRIPT = "run_crawler.py"
PIPELINE = 1  # Pipeline 1: Fast metadata scraping

# Scraper arguments
ACCOUNT_LIST = "list100"  # Name of list in crawl_account/ (without .txt)
# OR use direct file path:
# ACCOUNT_FILE = "crawl_account/list100.txt"

LOOKBACK_DAYS = 30  # Scrape videos from last N days
MAX_VIDEOS = 1000   # Max videos per profile
MAX_PROFILES = None  # Limit profiles (None = all)

# Display settings (for Linux servers)
DISPLAY = ":2"  # Set to None if not needed
HEADLESS = True  # Run in headless mode
USE_XVFB = False  # Use Xvfb for virtual display

# Proxy settings (optional)
PROXY = None  # Format: "ip:port:user:pass" or "http://ip:port"

# Advanced settings
RESTART_BROWSER_EVERY = 10  # Restart browser every N profiles
VERBOSE = False  # Verbose logging

# Maximum time to allow scraper to run (in seconds)
# Set to ~90% of INTERVAL_SECONDS to leave buffer
MAX_RUN_TIME = int(INTERVAL_SECONDS * 0.9)  # e.g., 54 minutes for 1 hour interval

# Logging
LOG_DIR = Path("logs/scheduler")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Stats tracking
STATS_FILE = LOG_DIR / f"scheduler_p{PIPELINE}_stats.json"


# ============================================================================
# BUILD COMMAND
# ============================================================================

def build_scraper_command():
    """Build the command to run the scraper"""
    cmd = [sys.executable, SCRAPER_SCRIPT]
    
    # Pipeline selection
    cmd.extend(["--pipeline", str(PIPELINE)])
    
    # Account list
    if 'ACCOUNT_FILE' in globals() and ACCOUNT_FILE:
        cmd.extend(["--account-file", ACCOUNT_FILE])
    else:
        cmd.extend(["--list", ACCOUNT_LIST])
    
    # Limits
    cmd.extend(["--lookback", str(LOOKBACK_DAYS)])
    cmd.extend(["--max-videos", str(MAX_VIDEOS)])
    
    if MAX_PROFILES:
        cmd.extend(["--max-profiles", str(MAX_PROFILES)])
    
    # Display
    if DISPLAY:
        cmd.extend(["--display", DISPLAY])
    
    if HEADLESS:
        cmd.append("--headless")
    
    if USE_XVFB:
        cmd.append("--xvfb")
    
    # Proxy
    if PROXY:
        cmd.extend(["--proxy", PROXY])
    
    # Advanced
    if RESTART_BROWSER_EVERY:
        cmd.extend(["--restart-every", str(RESTART_BROWSER_EVERY)])
    
    if VERBOSE:
        cmd.append("--verbose")
    
    return cmd


# ============================================================================
# TIMEOUT HANDLER
# ============================================================================

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Scraper exceeded maximum run time")


# ============================================================================
# STATS TRACKING
# ============================================================================

def load_stats():
    """Load scheduler statistics"""
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "pipeline": PIPELINE,
        "interval_seconds": INTERVAL_SECONDS,
        "total_runs": 0,
        "successful_runs": 0,
        "failed_runs": 0,
        "timeout_runs": 0,
        "runs": []
    }

def save_stats(stats):
    """Save scheduler statistics"""
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save stats: {e}")

def add_run_record(stats, start_time, end_time, status, error=None):
    """Add a run record to stats"""
    duration = (end_time - start_time).total_seconds()
    
    record = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "duration_minutes": round(duration / 60, 2),
        "status": status,
        "error": error
    }
    
    stats["runs"].append(record)
    stats["total_runs"] += 1
    
    if status == "success":
        stats["successful_runs"] += 1
    elif status == "timeout":
        stats["timeout_runs"] += 1
    else:
        stats["failed_runs"] += 1
    
    # Keep only last 100 runs
    if len(stats["runs"]) > 100:
        stats["runs"] = stats["runs"][-100:]
    
    return stats


# ============================================================================
# MAIN SCHEDULER
# ============================================================================

def run_scraper_with_timeout():
    """Run the scraper with timeout protection"""
    print(f"\n{'='*70}")
    print(f"Starting Pipeline {PIPELINE} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    
    try:
        # Build command
        cmd = build_scraper_command()
        print(f"Command: {' '.join(cmd)}\n")
        
        # Set up timeout alarm (Unix only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(MAX_RUN_TIME)
        
        # Run scraper
        result = subprocess.run(
            cmd,
            timeout=MAX_RUN_TIME,  # Backup timeout for Windows
            capture_output=False  # Let output go to console/log
        )
        
        # Cancel alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            print(f"\n✅ Scraper completed successfully in {duration:.1f}s ({duration/60:.1f} min)")
            return "success", None
        else:
            error = f"Exit code: {result.returncode}"
            print(f"\n❌ Scraper failed: {error}")
            return "failed", error
            
    except subprocess.TimeoutExpired:
        end_time = datetime.now()
        error = f"Timeout after {MAX_RUN_TIME}s"
        print(f"\n⏱️ Scraper timeout: {error}")
        return "timeout", error
        
    except TimeoutError as e:
        end_time = datetime.now()
        error = str(e)
        print(f"\n⏱️ {error}")
        return "timeout", error
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        raise
        
    except Exception as e:
        end_time = datetime.now()
        error = str(e)
        print(f"\n❌ Unexpected error: {error}")
        import traceback
        traceback.print_exc()
        return "failed", error
    
    finally:
        # Make sure to cancel alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def calculate_next_run_time(last_run_start, interval_seconds):
    """
    Calculate next run time to maintain consistent intervals.
    
    Instead of: last_end + interval (causes drift)
    Use: last_start + interval (consistent schedule)
    """
    next_run = last_run_start + timedelta(seconds=interval_seconds)
    
    # If we're already past the next run time, schedule immediately
    now = datetime.now()
    if next_run < now:
        print(f"⚠️ Warning: Fell behind schedule. Running immediately.")
        return now
    
    return next_run


def format_time_remaining(seconds):
    """Format seconds into human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.1f}h {minutes:.0f}m"


def main():
    print("="*70)
    print(f"TikTok Pipeline {PIPELINE} Scheduler")
    print("="*70)
    print(f"Pipeline: {PIPELINE} (Fast Metadata Scraping)")
    print(f"Interval: {INTERVAL_SECONDS}s ({format_time_remaining(INTERVAL_SECONDS)})")
    print(f"Max run time: {MAX_RUN_TIME}s ({format_time_remaining(MAX_RUN_TIME)})")
    print(f"Lookback: {LOOKBACK_DAYS} days")
    print(f"Account list: {ACCOUNT_LIST if 'ACCOUNT_FILE' not in globals() else ACCOUNT_FILE}")
    if MAX_PROFILES:
        print(f"Max profiles: {MAX_PROFILES}")
    print("="*70)
    
    # Validate script exists
    if not Path(SCRAPER_SCRIPT).exists():
        print(f"\n❌ Error: Scraper script not found: {SCRAPER_SCRIPT}")
        sys.exit(1)
    
    # Load stats
    stats = load_stats()
    
    # Print previous stats if available
    if stats["total_runs"] > 0:
        success_rate = (stats['successful_runs'] / stats['total_runs']) * 100
        print(f"\nPrevious runs: {stats['total_runs']}")
        print(f"  ✅ Success: {stats['successful_runs']} ({success_rate:.1f}%)")
        print(f"  ❌ Failed: {stats['failed_runs']}")
        print(f"  ⏱️  Timeout: {stats['timeout_runs']}")
        
        # Show last run
        if stats["runs"]:
            last_run = stats["runs"][-1]
            print(f"\nLast run:")
            print(f"  Time: {last_run['start_time']}")
            print(f"  Duration: {last_run['duration_minutes']:.1f} min")
            print(f"  Status: {last_run['status']}")
    
    print(f"\n💡 Press Ctrl+C to stop scheduler\n")
    print(f"Stats will be saved to: {STATS_FILE}\n")
    
    try:
        run_number = stats["total_runs"] + 1
        
        while True:
            run_start = datetime.now()
            
            print(f"\n{'#'*70}")
            print(f"# RUN #{run_number}")
            print(f"{'#'*70}")
            
            # Run scraper
            status, error = run_scraper_with_timeout()
            
            run_end = datetime.now()
            duration = (run_end - run_start).total_seconds()
            
            # Update stats
            stats = add_run_record(stats, run_start, run_end, status, error)
            save_stats(stats)
            
            # Calculate next run time (consistent intervals)
            next_run_time = calculate_next_run_time(run_start, INTERVAL_SECONDS)
            now = datetime.now()
            
            sleep_seconds = (next_run_time - now).total_seconds()
            
            if sleep_seconds > 0:
                success_rate = (stats['successful_runs'] / stats['total_runs']) * 100
                
                print(f"\n{'='*70}")
                print(f"Run #{run_number} completed in {duration/60:.1f} minutes")
                print(f"Status: {status.upper()}")
                print(f"{'='*70}")
                print(f"Next run: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Sleep time: {format_time_remaining(sleep_seconds)}")
                print(f"{'='*70}")
                print(f"Session stats: {stats['successful_runs']}/{stats['total_runs']} successful ({success_rate:.1f}%)")
                print(f"{'='*70}\n")
                
                time.sleep(sleep_seconds)
            else:
                print(f"\n⚠️ Warning: No sleep time (already {abs(sleep_seconds):.1f}s behind)")
                print(f"Consider increasing interval or reducing max profiles\n")
            
            run_number += 1
    
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("🛑 SCHEDULER STOPPED BY USER")
        print("="*70)
        
        if stats['total_runs'] > 0:
            success_rate = (stats['successful_runs'] / stats['total_runs']) * 100
            avg_duration = sum(r['duration_seconds'] for r in stats['runs']) / len(stats['runs'])
            
            print(f"\nFinal Statistics:")
            print(f"  Total runs: {stats['total_runs']}")
            print(f"  ✅ Successful: {stats['successful_runs']} ({success_rate:.1f}%)")
            print(f"  ❌ Failed: {stats['failed_runs']}")
            print(f"  ⏱️  Timeout: {stats['timeout_runs']}")
            print(f"  📊 Avg duration: {avg_duration/60:.1f} min")
            print(f"\nStats saved to: {STATS_FILE}")
        
        print("="*70 + "\n")
        sys.exit(0)


if __name__ == "__main__":
    # Validate configuration
    if not Path(SCRAPER_SCRIPT).exists():
        print(f"❌ Error: Scraper script not found: {SCRAPER_SCRIPT}")
        print(f"   Make sure {SCRAPER_SCRIPT} is in the current directory")
        sys.exit(1)
    
    # Warn if max run time is too close to interval
    if MAX_RUN_TIME >= INTERVAL_SECONDS:
        print(f"\n⚠️  WARNING: MAX_RUN_TIME ({MAX_RUN_TIME}s) >= INTERVAL ({INTERVAL_SECONDS}s)")
        print(f"   This may cause overlapping runs!")
        print(f"   Recommended: MAX_RUN_TIME < {INTERVAL_SECONDS * 0.9:.0f}s (90% of interval)\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    main()