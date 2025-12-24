"""
TikTok Profile Time Series Research Runner

Scrapes TikTok user profiles and creates time series datasets showing
engagement metrics over time (daily/weekly/monthly aggregation).

Usage:
    python run_crawler.py                    # Scrape all profiles, daily aggregation
    python run_crawler.py --no-resume        # Rerun scrape for scraped profiles
    python run_crawler.py --test             # Test mode (2 profiles)
    python run_crawler.py --lookback 180     # Last 6 months instead of 1 year
    python run_crawler.py --max-profiles 50  # Limit to 50 profiles
"""

import asyncio
import json
import logging
import sys
import os
import subprocess
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Add parent directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from TT_Content_Scraper.src.scraper_functions.playwright_scraper import (
    PlaywrightProfileScraper, VideoData, TimeSeriesPoint
)
from TT_Content_Scraper.src.object_tracker_db import ObjectTracker, ObjectStatus


# ============================================================================
# CONFIGURATION
# ============================================================================

# Account ID file
ACCOUNT_ID_FILE = "dataset/account_ids.txt"

# Output settings
OUTPUT_DIR = "data/profile_time_series/"
PROGRESS_DB_DIR = "progress_tracking/"

# Scraping parameters
DEFAULT_LOOKBACK_DAYS = 180
DEFAULT_MAX_VIDEOS = 10000       # Max videos per profile
PROFILE_DELAY = 2.0              # Delay between profiles (seconds)

# Browser settings
IS_HEADLESS_SERVER = os.environ.get('DISPLAY') is None and sys.platform.startswith('linux')
HEADLESS = IS_HEADLESS_SERVER
USE_XVFB = False
SLOW_MO = 50

# Hardcoded proxy (set your proxy here)
PROXY = "14.224.198.119:44182:HcgsFh:ZnHhhU"


# ============================================================================
# XVFB PSEUDO-HEADLESS MODE (Ubuntu/Linux)
# ============================================================================

class XvfbDisplay:
    """
    Manage Xvfb (X Virtual Framebuffer) for pseudo-headless mode.
    
    Requirements:
        sudo apt-get install xvfb x11-utils
    """
    
    def __init__(self, display: str = ':99', screen: str = '1920x1080x24'):
        self.display = display
        self.screen = screen
        self.process = None
        self.logger = logging.getLogger('XvfbDisplay')
        self.original_display = os.environ.get('DISPLAY')
    
    def start(self) -> bool:
        """Start Xvfb display server"""
        try:
            subprocess.run(['which', 'Xvfb'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("Xvfb not installed. Install with: sudo apt-get install xvfb x11-utils")
            return False
        
        try:
            self.process = subprocess.Popen(
                ['Xvfb', self.display, '-screen', '0', self.screen, '-ac', '-noreset'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            
            # Wait for display to be ready
            for i in range(10):
                try:
                    result = subprocess.run(
                        ['xdpyinfo', '-display', self.display],
                        capture_output=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        os.environ['DISPLAY'] = self.display
                        self.logger.info(f"Xvfb started on {self.display}")
                        time.sleep(random.uniform(0.5, 2.0))
                        return True
                except Exception:
                    pass
                time.sleep(0.5 if i < 5 else 1.0)
            
            self.logger.error("Xvfb failed to start")
            if self.process:
                self.process.terminate()
            return False
        
        except Exception as e:
            self.logger.error(f"Failed to start Xvfb: {e}")
            return False
    
    def stop(self):
        """Stop Xvfb display server"""
        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.logger.info("Xvfb stopped")
            except Exception as e:
                self.logger.warning(f"Error stopping Xvfb: {e}")
                try:
                    self.process.kill()
                except:
                    pass
        
        if self.original_display:
            os.environ['DISPLAY'] = self.original_display
        elif 'DISPLAY' in os.environ:
            del os.environ['DISPLAY']
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProfileTracker:
    """
    Track profile scraping progress using SQLite.
    
    Allows resuming if interrupted.
    """
    
    def __init__(self, run_name: str, db_dir: str = PROGRESS_DB_DIR):
        self.run_name = run_name
        db_path = Path(db_dir) / f"{run_name}_progress.db"
        self.tracker = ObjectTracker(db_file=str(db_path))
        self.logger = logging.getLogger('ProfileTracker')
    
    def add_profiles(self, account_ids: List[str]):
        """Add account IDs to track (pending status)"""
        for account_id in account_ids:
            self.tracker.add_object(
                id=account_id,
                title=f"account_{account_id}",
                type="profile"
            )
        self.logger.info(f"Added {len(account_ids)} profiles to tracker")
    
    def get_pending_profiles(self) -> List[str]:
        """Get profiles that haven't been completed yet"""
        pending = self.tracker.get_pending_objects(type="profile")
        return list(pending.keys())
    
    def get_completed_profiles(self) -> List[str]:
        """Get profiles that have been completed"""
        completed = self.tracker.get_completed_objects()
        return [p for p, info in completed.items() if info.get('type') == 'profile']
    
    def mark_completed(self, account_id: str, videos: int, time_points: int):
        """Mark profile as completed with stats"""
        file_path = f"videos={videos}, time_series_points={time_points}"
        self.tracker.mark_completed(account_id, file_path=file_path)
        self.logger.debug(f"Marked account {account_id} as completed")
    
    def mark_error(self, account_id: str, error: str):
        """Mark profile as error"""
        self.tracker.mark_error(account_id, error)
        self.logger.warning(f"Marked account {account_id} as error: {error}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get progress statistics"""
        stats = self.tracker.get_stats()
        return {
            "total": stats.get("total", 0),
            "completed": stats.get("completed", 0),
            "pending": stats.get("pending", 0),
            "error": stats.get("error", 0)
        }
    
    def is_completed(self, account_id: str) -> bool:
        """Check if profile was already completed"""
        return account_id in self.get_completed_profiles()
    
    def close(self):
        """Close database connection"""
        if self.tracker.conn:
            self.tracker.conn.close()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_account_ids(file_path: str) -> List[str]:
    """Load account IDs from a text file"""
    account_ids = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Warning: Account ID file not found: {file_path}")
        return account_ids
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            account_ids.append(line)
    
    return account_ids


def setup_logging(run_name: str, log_dir: str = "logs/", verbose: bool = False) -> logging.Logger:
    """Set up logging for a run"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler: DEBUG level (detailed logs)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Console handler: DEBUG if verbose, INFO otherwise
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Set root logger to DEBUG so file handler captures everything
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Reduce playwright logging noise
    logging.getLogger('playwright').setLevel(logging.WARNING)
    
    return logging.getLogger('ProfileTimeSeries')


# ============================================================================
# MAIN SCRAPING LOGIC
# ============================================================================

async def scrape_profile_time_series(
    account_ids: List[str],
    lookback_days: int = 365,
    max_videos: int = 1000,
    resume: bool = True,
    verbose: bool = False,
    logger = None
) -> Dict[str, Any]:
    """
    Scrape time series data for multiple TikTok profiles.
    
    Args:
        account_ids: List of account IDs or usernames
        lookback_days: How many days back to scrape
        max_videos: Max videos per profile
        resume: Skip already completed profiles
        logger: Logger instance
    
    Returns:
        Summary of scraping results
    """
    if logger is None:
        logger = setup_logging("profile_timeseries", verbose=verbose)
    
    logger.info("=" * 70)
    logger.info("TIKTOK PROFILE TIME SERIES SCRAPING")
    logger.info(f"Lookback: {lookback_days} days")
    logger.info(f"Max videos per profile: {max_videos}")
    logger.info(f"Total profiles: {len(account_ids)}")
    logger.info("=" * 70)
    
    # Initialize progress tracker
    run_name = f"timeseries_{datetime.now().strftime('%Y%m%d')}"
    tracker = ProfileTracker(run_name)
    
    # Add all profiles to tracker
    tracker.add_profiles(account_ids)
    
    # Filter to pending profiles if resume mode
    if resume:
        pending = tracker.get_pending_profiles()
        completed = tracker.get_completed_profiles()
        logger.info(f"Resume mode: {len(completed)} already completed, {len(pending)} pending")
        account_ids = pending
    
    if not account_ids:
        logger.info("No profiles to scrape!")
        tracker.close()
        return {"message": "All profiles already completed"}
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(OUTPUT_DIR) / f"{lookback_days}" / f"{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results tracking
    results = {
        "run_name": run_name,
        "lookback_days": lookback_days,
        "output_dir": str(output_dir),
        "started_at": datetime.now().isoformat(),
        "profiles_processed": 0,
        "profiles_success": 0,
        "profiles_failed": 0,
        "total_videos": 0,
        "total_time_points": 0,
        "profile_summaries": {}
    }
    
    # Start scraping
    async with PlaywrightProfileScraper(
        output_dir=str(output_dir),
        headless=HEADLESS,
        slow_mo=SLOW_MO,
        wait_time=1.0,
        proxy=PROXY
    ) as scraper:
        
        for i, account_id in enumerate(account_ids):
            logger.info(f"\n[{i+1}/{len(account_ids)}] Processing: {account_id}")
            
            try:
                # Delay between profiles (except first one)
                if i > 0:
                    await asyncio.sleep(PROFILE_DELAY)
                
                # Scrape user time series
                # Note: If account_id looks like a username (alphanumeric), 
                # use it directly. Otherwise, need by_id=True
                is_username = account_id.replace('_', '').replace('.', '').isalnum()
                
                summary = await scraper.scrape_user_time_series(
                    username=account_id if is_username else None,
                    max_videos=max_videos,
                    lookback_days=lookback_days
                )
                
                if "error" in summary:
                    logger.warning(f"  Failed: {summary['error']}")
                    tracker.mark_error(account_id, summary['error'])
                    results["profiles_failed"] += 1
                else:
                    # Success
                    videos = summary.get('total_videos', 0)
                    time_points = summary.get('time_series_points', 0)
                    
                    logger.info(f"  ✓ Success: {videos} videos, {time_points} time points")
                    
                    tracker.mark_completed(account_id, videos, time_points)
                    results["profiles_success"] += 1
                    results["total_videos"] += videos
                    results["total_time_points"] += time_points
                    
                    # Store summary
                    results["profile_summaries"][account_id] = {
                        "username": summary.get("username"),
                        "videos": videos,
                        "time_points": time_points,
                        "date_range": summary.get("date_range"),
                        "files": summary.get("files")
                    }
                
                results["profiles_processed"] += 1
                
                # Log progress every 10 profiles
                if (i + 1) % 10 == 0:
                    progress = tracker.get_stats()
                    logger.info(f"\n{'='*40}")
                    logger.info(f"Progress: {progress['completed']}/{progress['total']} completed")
                    logger.info(f"Success: {results['profiles_success']}, Failed: {results['profiles_failed']}")
                    logger.info(f"{'='*40}\n")
                
            except Exception as e:
                logger.error(f"  Error: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                tracker.mark_error(account_id, str(e))
                results["profiles_failed"] += 1
                results["profiles_processed"] += 1
    
    # Finalize
    results["completed_at"] = datetime.now().isoformat()
    results["scraper_stats"] = scraper.stats
    
    # Save summary
    summary_file = output_dir / f"run_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Close tracker
    tracker.close()
    
    # Log final summary
    logger.info("\n" + "=" * 70)
    logger.info("SCRAPING COMPLETED!")
    logger.info("=" * 70)
    logger.info(f"Processed: {results['profiles_processed']}")
    logger.info(f"Success: {results['profiles_success']}")
    logger.info(f"Failed: {results['profiles_failed']}")
    logger.info(f"Total videos: {results['total_videos']}")
    logger.info(f"Total time points: {results['total_time_points']}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Summary: {summary_file}")
    logger.info("=" * 70)
    
    return results


# ============================================================================
# CLI HELPERS
# ============================================================================

def show_progress_status():
    """Show progress status for recent runs"""
    print(f"\n{'='*60}")
    print("PROFILE SCRAPING PROGRESS STATUS")
    print(f"{'='*60}")
    
    db_dir = Path(PROGRESS_DB_DIR)
    if not db_dir.exists():
        print("No progress data found")
        return
    
    db_files = list(db_dir.glob("timeseries_*_progress.db"))
    
    if not db_files:
        print("No active runs found")
        return
    
    for db_file in sorted(db_files, reverse=True)[:5]:  # Show last 5 runs
        run_name = db_file.stem.replace('_progress', '')
        print(f"\nRun: {run_name}")
        
        try:
            tracker = ProfileTracker(run_name)
            stats = tracker.get_stats()
            tracker.close()
            
            print(f"  Total: {stats['total']}")
            print(f"  Completed: {stats['completed']}")
            print(f"  Pending: {stats['pending']}")
            print(f"  Error: {stats['error']}")
            
            if stats['total'] > 0:
                completion = (stats['completed'] / stats['total']) * 100
                print(f"  Progress: {completion:.1f}%")
        except Exception as e:
            print(f"  Error reading: {e}")
    
    print(f"\n{'='*60}")


def reset_progress(run_name: str = None):
    """Reset progress for a specific run or all runs"""
    db_dir = Path(PROGRESS_DB_DIR)
    
    if run_name:
        db_path = db_dir / f"{run_name}_progress.db"
        if db_path.exists():
            db_path.unlink()
            print(f"Reset progress for: {run_name}")
        else:
            print(f"No progress file found: {run_name}")
    else:
        # Reset all
        db_files = list(db_dir.glob("*_progress.db"))
        for db_file in db_files:
            db_file.unlink()
        print(f"Reset {len(db_files)} progress files")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TikTok Profile Time Series Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python run_crawler.py --lookback 180     # Last 6 months
            python run_crawler.py --test             # Test mode (2 profiles)
            python run_crawler.py --status           # Show progress
            python run_crawler.py --reset            # Reset all progress
            python run_crawler.py --no-resume        # Start fresh
        """
    )
    
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help="Days to look back (default: 180 days)")
    parser.add_argument("--max-videos", type=int, default=DEFAULT_MAX_VIDEOS,
                        help="Max videos per profile")
    parser.add_argument("--max-profiles", type=int, default=None,
                        help="Limit number of profiles to scrape")
    parser.add_argument("--account-file", type=str, default=ACCOUNT_ID_FILE,
                        help="Path to account IDs file")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh (don't skip completed profiles)")
    parser.add_argument("--status", action="store_true",
                        help="Show progress status")
    parser.add_argument("--reset", action="store_true",
                        help="Reset progress tracking")
    parser.add_argument("--proxy", type=str, default=None,
                        help="Proxy URL (overrides hardcoded proxy)")
    parser.add_argument("--headless", action="store_true",
                        help="Force headless mode")
    parser.add_argument("--no-headless", action="store_true",
                        help="Force non-headless mode")
    parser.add_argument("--xvfb", action="store_true",
                        help="Use Xvfb pseudo-headless (Linux only)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode (2 profiles)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose (DEBUG) console output")
    
    args = parser.parse_args()
    
    # Handle status command
    if args.status:
        show_progress_status()
        return
    
    # Handle reset command
    if args.reset:
        reset_progress()
        return
    
    # Handle headless mode
    headless_mode = HEADLESS
    if args.headless:
        headless_mode = True
    elif args.no_headless:
        headless_mode = False
    
    # Initialize Xvfb if requested
    xvfb = None
    if args.xvfb:
        if not sys.platform.startswith('linux'):
            print("ERROR: Xvfb is only available on Linux")
            return
        
        print("Starting Xvfb pseudo-headless display...")
        xvfb = XvfbDisplay()
        if not xvfb.start():
            print("ERROR: Failed to start Xvfb")
            return
        
        headless_mode = False
        print("Xvfb started - using pseudo-headless mode")
    
    # Set global settings
    if args.headless or args.no_headless:
        globals()['HEADLESS'] = headless_mode
    
    if args.proxy:
        globals()['PROXY'] = args.proxy
    
    # Load account IDs
    account_ids = load_account_ids(args.account_file)
    
    if not account_ids:
        print(f"ERROR: No account IDs loaded from {args.account_file}")
        if xvfb:
            xvfb.stop()
        return
    
    # Apply limits
    if args.test:
        account_ids = account_ids[:2]
    elif args.max_profiles:
        account_ids = account_ids[:args.max_profiles]
    
    # Show configuration
    print(f"\n{'='*60}")
    print("TikTok Profile Time Series Scraper")
    print(f"{'='*60}")
    print(f"Profiles: {len(account_ids)}")
    print(f"Lookback: {args.lookback} days")
    print(f"Max videos: {args.max_videos}")
    print(f"Resume: {not args.no_resume}")
    print(f"Headless: {headless_mode}")
    print(f"Verbose: {args.verbose}")
    if args.xvfb:
        print(f"Display: Xvfb (pseudo-headless)")
    if args.proxy:
        print(f"Proxy: {args.proxy}")
    print(f"{'='*60}\n")
    
    try:
        # Run scraping
        asyncio.run(scrape_profile_time_series(
            account_ids=account_ids,
            lookback_days=args.lookback,
            max_videos=args.max_videos,
            resume=not args.no_resume,
            verbose=args.verbose
        ))
    finally:
        # Cleanup
        if xvfb:
            xvfb.stop()


if __name__ == "__main__":
    main()