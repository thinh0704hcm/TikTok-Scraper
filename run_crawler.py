"""
TikTok Profile Scraper Runner

Orchestrates the scraping of multiple profiles, managing:
1. Browser Lifecycle (Playwright + Stealth)
2. Display Management (Xvfb or VNC)
3. Progress Tracking (SQLite)
4. Milestone/Date Logic (File-based or Argument-based)
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

# Add parent directory to path to ensure imports work
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# ----------------------------------------------------------------------------
# IMPORTS: Handle different folder structures
# ----------------------------------------------------------------------------
try:
    from TT_Content_Scraper.src.scraper_functions.playwright_scraper import PlaywrightScraper
    from TT_Content_Scraper.src.object_tracker_db import ObjectTracker
except ImportError:
    try:
        from scraper_functions.playwright_scraper import PlaywrightScraper
        from object_tracker_db import ObjectTracker
    except ImportError:
        # Fallback if running locally in the same directory
        from playwright_scraper import PlaywrightScraper
        from object_tracker_db import ObjectTracker

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories
ACCOUNT_LIST_DIR = "crawl_account/"
OUTPUT_BASE_DIR = "video_data/"
PROGRESS_DB_DIR = "progress_tracking/"
MILESTONE_FILE = "milestone_datetime.txt"

# Default Scraping parameters
DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_MAX_VIDEOS = 1000      # Safety limit
PROFILE_DELAY = 2.0            # Delay between profiles (seconds)

# Browser / Proxy
SLOW_MO = 50
PROXY = "14.224.198.119:44182:HcgsFh:ZnHhhU"
HEADLESS = True 

# ============================================================================
# DISPLAY MANAGEMENT (Xvfb + VNC Support)
# ============================================================================

class DisplayManager:
    """Manages display for headless environments (Xvfb or VNC)"""
    
    def __init__(self, display: str = None, use_xvfb: bool = False, screen: str = '1920x1080x24'):
        self.display = display or os.environ.get('DISPLAY', ':99')
        self.screen = screen
        self.use_xvfb = use_xvfb
        self.process = None
        self.logger = logging.getLogger('DisplayManager')
        self.original_display = os.environ.get('DISPLAY')
    
    def start(self) -> bool:
        """Start display (Xvfb if requested, otherwise use existing display)"""
        
        # If display is explicitly provided (like :2 for VNC), use it directly
        if self.display and not self.use_xvfb:
            os.environ['DISPLAY'] = self.display
            self.logger.info(f"✓ Using existing display: {self.display}")
            return True
        
        # Start Xvfb if requested
        if self.use_xvfb:
            return self._start_xvfb()
        
        return True
    
    def _start_xvfb(self) -> bool:
        """Start Xvfb virtual display"""
        try:
            subprocess.run(['which', 'Xvfb'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("Xvfb not installed. Install with: sudo apt-get install xvfb x11-utils")
            return False
        
        try:
            self.process = subprocess.Popen(
                ['Xvfb', self.display, '-screen', '0', self.screen, '-ac', '-noreset'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            # Wait for Xvfb to spin up
            time.sleep(2)
            os.environ['DISPLAY'] = self.display
            self.logger.info(f"✓ Xvfb started on {self.display}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start Xvfb: {e}")
            return False
    
    def stop(self):
        """Stop display (only stops Xvfb if it was started)"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                pass
        
        # Restore original display
        if self.original_display:
            os.environ['DISPLAY'] = self.original_display

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProfileTracker:
    def __init__(self, run_name: str, db_dir: str = PROGRESS_DB_DIR):
        self.run_name = run_name
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        db_path = Path(db_dir) / f"{run_name}_progress.db"
        self.tracker = ObjectTracker(db_file=str(db_path))
        self.logger = logging.getLogger('ProfileTracker')
    
    def add_profiles(self, account_ids: List[str]):
        for account_id in account_ids:
            self.tracker.add_object(id=account_id, title=f"account_{account_id}", type="profile")
    
    def get_pending_profiles(self) -> List[str]:
        pending = self.tracker.get_pending_objects(type="profile")
        return list(pending.keys())
    
    def get_completed_profiles(self) -> List[str]:
        completed = self.tracker.get_completed_objects()
        return [p for p, info in completed.items() if info.get('type') == 'profile']
    
    def mark_completed(self, account_id: str, videos: int):
        file_path = f"videos={videos}"
        self.tracker.mark_completed(account_id, file_path=file_path)
    
    def mark_error(self, account_id: str, error: str):
        self.tracker.mark_error(account_id, error)
    
    def get_stats(self) -> Dict[str, int]:
        stats = self.tracker.get_stats()
        return {
            "total": stats.get("total", 0),
            "completed": stats.get("completed", 0),
            "pending": stats.get("pending", 0),
            "error": stats.get("error", 0)
        }
    
    def close(self):
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

def read_milestone_datetime(file_path: str = MILESTONE_FILE) -> Optional[datetime]:
    """Read milestone datetime from text file (first line)"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                return None
            
            # Try parsing datetime (YYYY-MM-DD HH:MM:SS)
            try:
                return datetime.strptime(first_line, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                # Try parsing just date (YYYY-MM-DD)
                return datetime.strptime(first_line, "%Y-%m-%d")
                
    except Exception as e:
        print(f"Error reading milestone file: {e}")
        return None

def setup_logging(run_name: str, log_dir: str = "logs/", verbose: bool = False) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logging.getLogger('playwright').setLevel(logging.WARNING)
    
    return logging.getLogger('Crawler')


# ============================================================================
# MAIN SCRAPING LOGIC
# ============================================================================

async def scrape_profiles(
    account_ids: List[str],
    milestone_dt: Optional[datetime],
    max_videos: int = 1000,
    resume: bool = True,
    verbose: bool = False,
    headless: bool = True,
    proxy: Optional[str] = None,
    list_name: str = "list32"
) -> Dict[str, Any]:
    
    logger = setup_logging(f"scrape_{list_name}", verbose=verbose)
    
    logger.info("=" * 70)
    logger.info("TIKTOK PROFILE SCRAPER")
    logger.info(f"Target Profiles: {len(account_ids)}")
    if milestone_dt:
        logger.info(f"Milestone Date: {milestone_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        logger.info("Milestone: None (Scraping all videos)")
    logger.info("=" * 70)
    
    # Initialize Tracker
    run_name = f"run_{list_name}_{datetime.now().strftime('%Y%m%d')}"
    tracker = ProfileTracker(run_name)
    tracker.add_profiles(account_ids)
    
    if resume:
        pending = tracker.get_pending_profiles()
        completed = tracker.get_completed_profiles()
        logger.info(f"Resume: {len(completed)} done, {len(pending)} pending")
        account_ids = pending
    
    if not account_ids:
        logger.info("No profiles to scrape.")
        tracker.close()
        return {}

    # Setup Output Directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(OUTPUT_BASE_DIR) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {run_dir}")
    
    # Track run statistics
    run_stats = {
        "run_name": run_name,
        "milestone_datetime": milestone_dt.strftime('%Y-%m-%d %H:%M:%S') if milestone_dt else None,
        "output_dir": str(run_dir),
        "started_at": datetime.now().isoformat(),
        "profiles_processed": 0,
        "profiles_success": 0,
        "profiles_failed": 0,
        "total_videos": 0,
        "profile_summaries": {}
    }

    # Start Scraper
    async with PlaywrightScraper(
        headless=headless,
        slow_mo=SLOW_MO,
        wait_time=10.0,
        proxy=proxy,
        fingerprint_file='browser_fingerprint.json'
    ) as scraper:
        
        for i, account_id in enumerate(account_ids):
            logger.info(f"\n[{i+1}/{len(account_ids)}] Processing: {account_id}")
            run_stats["profiles_processed"] += 1
            
            try:
                if i > 0: 
                    await asyncio.sleep(PROFILE_DELAY)
                
                # Create profile directory
                profile_dir = run_dir / account_id
                
                # --- SCRAPE ---
                summary = await scraper.scrape_user_profile(
                    username=account_id,
                    profile_dir=profile_dir,
                    max_videos=max_videos,
                    milestone_datetime=milestone_dt
                )
                
                if "error" in summary:
                    logger.warning(f"  Failed: {summary['error']}")
                    tracker.mark_error(account_id, summary['error'])
                    run_stats["profiles_failed"] += 1
                else:
                    videos = summary.get('total_videos', 0)
                    
                    logger.info(f"  ✓ Success: {videos} videos")
                    tracker.mark_completed(account_id, videos)
                    run_stats["profiles_success"] += 1
                    run_stats["total_videos"] += videos
                    
                    # Add to profile summaries
                    run_stats["profile_summaries"][account_id] = {
                        "username": account_id,
                        "videos": videos,
                        "date_range": summary.get('date_range'),
                        "files": summary.get('files')
                    }
                
            except Exception as e:
                logger.error(f"  Critical Error: {e}")
                import traceback
                traceback.print_exc()
                tracker.mark_error(account_id, str(e))
                run_stats["profiles_failed"] += 1
    
    # Save run summary
    run_stats["completed_at"] = datetime.now().isoformat()
    run_stats["scraper_stats"] = scraper.stats
    
    run_summary_file = run_dir / f"run_summary_{timestamp}.json"
    with open(run_summary_file, 'w', encoding='utf-8') as f:
        json.dump(run_stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n✅ Run complete! Summary saved to {run_summary_file}")
    logger.info(f"   - Profiles processed: {run_stats['profiles_processed']}")
    logger.info(f"   - Success: {run_stats['profiles_success']}")
    logger.info(f"   - Failed: {run_stats['profiles_failed']}")
    logger.info(f"   - Total videos: {run_stats['total_videos']}")
    
    tracker.close()
    return run_stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="TikTok Profile Scraper")
    
    # Input/Output
    parser.add_argument("--list", type=str, default="list32", help="Name of list in crawl_account/")
    parser.add_argument("--account-file", type=str, default=None, help="Direct path to account file")
    
    # Limits & Logic
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Days to look back (if no milestone file)")
    parser.add_argument("--max-videos", type=int, default=DEFAULT_MAX_VIDEOS)
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--test", action="store_true", help="Run on first 2 profiles only")
    
    # Technical
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--xvfb", action="store_true")
    parser.add_argument("--display", type=str, default=None, help="Display to use (e.g. :2)")
    parser.add_argument("--proxy", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--status", action="store_true", help="Show progress of recent runs")
    parser.add_argument("--reset", action="store_true", help="Reset all progress")
    
    args = parser.parse_args()
    
    # Handle Maintenance Commands
    if args.reset:
        for f in Path(PROGRESS_DB_DIR).glob("*.db"): f.unlink()
        print("Progress reset.")
        return
    if args.status:
        print("Status check not implemented in this snippet.")
        return

    # 1. Determine Display/Headless Settings
    display_to_use = args.display
    use_xvfb = args.xvfb
    # If no display arg but we are on linux without display, default to Xvfb
    if not display_to_use and not use_xvfb and sys.platform == 'linux' and not os.environ.get('DISPLAY'):
        use_xvfb = True
        
    headless_mode = args.headless or HEADLESS
    # If using VNC (:2), we usually want headless=False to debug, or True to save resources
    if display_to_use: 
        headless_mode = False # Assume debug if explicitly setting display

    # 2. Determine Account List
    if args.account_file:
        account_file = args.account_file
        list_name = Path(account_file).stem
    else:
        list_name = args.list
        account_file = str(Path(ACCOUNT_LIST_DIR) / f"{list_name}.txt")
    
    account_ids = load_account_ids(account_file)
    if not account_ids:
        print(f"Error: No accounts found in {account_file}")
        return
        
    if args.test: 
        account_ids = account_ids[:2]
        print(f"Test mode: Running on first 2 profiles")
    elif args.max_profiles: 
        account_ids = account_ids[:args.max_profiles]

    # 3. Determine Milestone
    milestone_dt = read_milestone_datetime()
    if not milestone_dt and args.lookback > 0:
        milestone_dt = datetime.now() - timedelta(days=args.lookback)
        print(f"No milestone file found. Calculated from lookback ({args.lookback} days): {milestone_dt}")
    elif milestone_dt:
        print(f"Loaded milestone from file: {milestone_dt}")

    # 4. Run
    with DisplayManager(display=display_to_use, use_xvfb=use_xvfb) as disp:
        asyncio.run(scrape_profiles(
            account_ids=account_ids,
            milestone_dt=milestone_dt,
            max_videos=args.max_videos,
            resume=not args.no_resume,
            verbose=args.verbose,
            headless=headless_mode,
            proxy=args.proxy,
            list_name=list_name
        ))

if __name__ == "__main__":
    main()