"""
Scheduled TikTok Crawler with Consistent Intervals

Runs the crawler at fixed intervals with proper error handling,
logging, and consistent scheduling.
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
# INTERVAL_SECONDS = 3600  # 1 hour
INTERVAL_SECONDS = 1800  # 30 minutes

# Crawler script settings
CRAWLER_SCRIPT = "run_crawler.py"
CRAWLER_ARGS = [
    "--display", ":2",
    "--no-resume",
]

# Maximum time to allow crawler to run (in seconds)
MAX_RUN_TIME = 1000  # 16 minutes and 40 seconds (leave buffer before next run)

# Logging
LOG_DIR = Path("logs/scheduler")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Stats tracking
STATS_FILE = LOG_DIR / "scheduler_stats.json"


# ============================================================================
# TIMEOUT HANDLER
# ============================================================================

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Crawler exceeded maximum run time")


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

def run_crawler_with_timeout():
    """Run the crawler with timeout protection"""
    print(f"\n{'='*70}")
    print(f"Starting crawler at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    
    try:
        # Set up timeout alarm (Unix only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(MAX_RUN_TIME)
        
        # Run crawler
        result = subprocess.run(
            [sys.executable, CRAWLER_SCRIPT] + CRAWLER_ARGS,
            timeout=MAX_RUN_TIME,  # Backup timeout for Windows
            capture_output=False  # Let output go to console/log
        )
        
        # Cancel alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            print(f"\n✅ Crawler completed successfully in {duration:.1f}s")
            return "success", None
        else:
            error = f"Exit code: {result.returncode}"
            print(f"\n❌ Crawler failed: {error}")
            return "failed", error
            
    except subprocess.TimeoutExpired:
        end_time = datetime.now()
        error = f"Timeout after {MAX_RUN_TIME}s"
        print(f"\n⏱️ Crawler timeout: {error}")
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


def main():
    print("="*70)
    print("TikTok Crawler Scheduler")
    print(f"Interval: {INTERVAL_SECONDS}s ({INTERVAL_SECONDS/3600:.1f} hours)")
    print(f"Max run time: {MAX_RUN_TIME}s ({MAX_RUN_TIME/60:.1f} minutes)")
    print(f"Script: {CRAWLER_SCRIPT}")
    print(f"Args: {' '.join(CRAWLER_ARGS)}")
    print("="*70)
    
    # Load stats
    stats = load_stats()
    
    # Print previous stats if available
    if stats["total_runs"] > 0:
        print(f"\nPrevious runs: {stats['total_runs']}")
        print(f"  Success: {stats['successful_runs']}")
        print(f"  Failed: {stats['failed_runs']}")
        print(f"  Timeout: {stats['timeout_runs']}")
    
    print(f"\nPress Ctrl+C to stop\n")
    
    try:
        while True:
            run_start = datetime.now()
            
            # Run crawler
            status, error = run_crawler_with_timeout()
            
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
                print(f"\n{'='*70}")
                print(f"Run completed in {duration:.1f}s")
                print(f"Next run at: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Sleeping for: {sleep_seconds:.1f}s ({sleep_seconds/60:.1f} minutes)")
                print(f"Stats: {stats['successful_runs']}/{stats['total_runs']} successful")
                print(f"{'='*70}\n")
                
                time.sleep(sleep_seconds)
            else:
                print(f"\n⚠️ Warning: No sleep time (already {abs(sleep_seconds):.1f}s behind)")
                print(f"Consider increasing interval or reducing max profiles\n")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Scheduler stopped by user")
        print(f"\nFinal stats:")
        print(f"  Total runs: {stats['total_runs']}")
        print(f"  Successful: {stats['successful_runs']}")
        print(f"  Failed: {stats['failed_runs']}")
        print(f"  Timeout: {stats['timeout_runs']}")
        print(f"\nStats saved to: {STATS_FILE}")
        sys.exit(0)


if __name__ == "__main__":
    # Validate configuration
    if not Path(CRAWLER_SCRIPT).exists():
        print(f"Error: Crawler script not found: {CRAWLER_SCRIPT}")
        sys.exit(1)
    
    if MAX_RUN_TIME >= INTERVAL_SECONDS:
        print(f"Warning: MAX_RUN_TIME ({MAX_RUN_TIME}s) >= INTERVAL ({INTERVAL_SECONDS}s)")
        print(f"This may cause overlapping runs!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    main()