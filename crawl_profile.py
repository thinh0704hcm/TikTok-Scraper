"""
TikTok Profile Data Crawler

Scrapes TikTok user profile metadata including follower count, heart count, 
and total videos.

Usage:
    python crawl_profile.py                    # Scrape all profiles from list
    python crawl_profile.py --list list32      # Use specific account list
    python crawl_profile.py --no-resume        # Rerun scrape for scraped profiles
    python crawl_profile.py --test             # Test mode (2 profiles)
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
from dataclasses import dataclass, asdict

# Add parent directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from TT_Content_Scraper.src.scraper_functions.playwright_scraper import PlaywrightProfileScraper
from TT_Content_Scraper.src.object_tracker_db import ObjectTracker, ObjectStatus


# ============================================================================
# CONFIGURATION
# ============================================================================

# Account list directory
ACCOUNT_LIST_DIR = "crawl_account/"

# Output settings
OUTPUT_BASE_DIR = "profile_data/"
PROGRESS_DB_DIR = "progress_tracking/"

# Scraping parameters
PROFILE_DELAY = 2.0              # Delay between profiles (seconds)

# Browser settings
IS_HEADLESS_SERVER = os.environ.get('DISPLAY') is None and sys.platform.startswith('linux')
HEADLESS = IS_HEADLESS_SERVER
USE_XVFB = False
SLOW_MO = 50

# Hardcoded proxy (set your proxy here)
PROXY = "14.224.198.119:44182:HcgsFh:ZnHhhU"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ProfileData:
    """Data class for profile information"""
    username: str
    follower_count: int
    heart_count: int
    total_videos: Optional[int]
    scraped_at: str


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
    
    def mark_completed(self, account_id: str, file_path: str = ""):
        """Mark profile as completed"""
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
# PROFILE SCRAPER
# ============================================================================

async def get_profile_data(
    scraper: PlaywrightProfileScraper,
    username: str,
    logger
) -> Optional[ProfileData]:
    """
    Extract profile metadata from TikTok user page.
    
    Args:
        scraper: Playwright scraper instance
        username: TikTok username (without @)
        logger: Logger instance
    
    Returns:
        ProfileData object with follower count, heart count, and total videos
    """
    url = f"https://www.tiktok.com/@{username}"
    
    try:
        logger.info(f"Navigating to @{username}...")
        
        # Navigate to profile page
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await scraper.page.goto(url, wait_until='domcontentloaded', timeout=30000)
                break
            except Exception as nav_err:
                if attempt < max_retries - 1:
                    wait_time = 3 * (attempt + 1)
                    logger.warning(f"Navigation attempt {attempt+1} failed, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        await asyncio.sleep(scraper.wait_time + 1)
        
        # Try to extract profile stats directly from the page DOM
        logger.debug("Attempting to extract profile stats from page DOM...")
        
        profile_stats = await scraper.page.evaluate('''
            () => {
                // Try to extract stats from DOM elements
                const result = {};
                
                // Method 1: Try data-e2e attributes (common in TikTok)
                const followerEl = document.querySelector('[data-e2e="followers-count"]');
                const heartEl = document.querySelector('[data-e2e="likes-count"]');
                
                if (followerEl) {
                    result.followerText = followerEl.textContent || followerEl.innerText;
                }
                if (heartEl) {
                    result.heartText = heartEl.textContent || heartEl.innerText;
                }
                
                // Method 2: Try to get from window data objects
                let data = window.__UNIVERSAL_DATA_FOR_REHYDRATION__ || window.SIGI_STATE;
                
                if (data && data.__DEFAULT_SCOPE__) {
                    const userDetail = data.__DEFAULT_SCOPE__['webapp.user-detail'];
                    if (userDetail && userDetail.userInfo && userDetail.userInfo.stats) {
                        const stats = userDetail.userInfo.stats;
                        result.followerCount = stats.followerCount;
                        result.heartCount = stats.heartCount;
                    }
                }
                
                // Method 3: Try UserModule
                if (data && data.UserModule && (!result.followerCount)) {
                    const users = Object.values(data.UserModule);
                    if (users.length > 0 && users[0].stats) {
                        const stats = users[0].stats;
                        result.followerCount = stats.followerCount;
                        result.heartCount = stats.heartCount;
                    }
                }
                
                return result;
            }
        ''')
        
        logger.debug(f"Extracted profile stats from DOM: {profile_stats}")
        
        if not profile_stats:
            logger.error(f"No profile stats extracted for @{username}")
            return None
        
        # Parse follower count
        follower_count = profile_stats.get('followerCount')
        if follower_count is None and profile_stats.get('followerText'):
            follower_count = parse_count_string(profile_stats.get('followerText', ''))
        
        # Parse heart count
        heart_count = profile_stats.get('heartCount')
        if heart_count is None and profile_stats.get('heartText'):
            heart_count = parse_count_string(profile_stats.get('heartText', ''))
        
        if follower_count is None:
            logger.error(f"Could not extract follower_count for @{username}")
            return None
        
        # Build result (without total_videos as per user request)
        result = {
            'scraped_at': datetime.now().isoformat(),
            'follower_count': follower_count,
            'heart_count': heart_count if heart_count is not None else 0
        }
        
        logger.info(f"Profile data for @{username}: followers={result.get('follower_count')}, hearts={result.get('heart_count')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting profile data for @{username}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_count_string(text: str) -> Optional[int]:
    """
    Parse count string like '1.2M', '345.6K', '1234' to integer.
    
    Args:
        text: Count string (e.g., '1.2M', '345K', '1234')
    
    Returns:
        Integer count or None
    """
    if not text:
        return None
        
    text = text.strip().upper()
    
    try:
        if 'M' in text:
            return int(float(text.replace('M', '')) * 1_000_000)
        elif 'K' in text:
            return int(float(text.replace('K', '')) * 1_000)
        else:
            # Remove any commas and parse
            return int(text.replace(',', ''))
    except:
        return None


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
    
    return logging.getLogger('ProfileDataScraper')


# ============================================================================
# MAIN SCRAPING LOGIC
# ============================================================================

async def scrape_profile_data(
    account_ids: List[str],
    resume: bool = True,
    verbose: bool = False,
    logger = None,
    list_name: str = "default"
) -> Dict[str, Any]:
    """
    Scrape profile data for multiple TikTok profiles.
    
    Args:
        account_ids: List of account IDs or usernames
        resume: Skip already completed profiles
        verbose: Verbose logging
        logger: Logger instance
        list_name: Name of the account list
    
    Returns:
        Summary of scraping results
    """
    if logger is None:
        logger = setup_logging("profile_data", verbose=verbose)
    
    logger.info("=" * 70)
    logger.info("TIKTOK PROFILE DATA SCRAPING")
    logger.info(f"Total profiles: {len(account_ids)}")
    logger.info("=" * 70)
    
    # Initialize progress tracker
    run_name = f"profile_{list_name}_{datetime.now().strftime('%Y%m%d')}"
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
    output_dir = Path(OUTPUT_BASE_DIR) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results tracking
    results = {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "started_at": datetime.now().isoformat(),
        "profiles_processed": 0,
        "profiles_success": 0,
        "profiles_failed": 0,
        "errors": []
    }
    
    # Initialize scraper
    logger.info("Initializing browser...")
    scraper = PlaywrightProfileScraper(
        headless=HEADLESS,
        slow_mo=SLOW_MO,
        wait_time=1.0,
        proxy=PROXY
    )
    
    try:
        await scraper.start()
        logger.info("Browser started successfully")
        
        # Scrape each profile
        for i, username in enumerate(account_ids, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Profile {i}/{len(account_ids)}: @{username}")
            logger.info(f"{'='*70}")
            
            try:
                # Get profile data
                profile_data = await get_profile_data(scraper, username, logger)
                
                if profile_data:
                    # Create username directory
                    username_dir = output_dir / username
                    username_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save profile data (profile_data is already a dict)
                    profile_file = username_dir / f"profile_{timestamp}.json"
                    with open(profile_file, 'w', encoding='utf-8') as f:
                        json.dump(profile_data, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"Saved profile data to {profile_file}")
                    
                    # Mark as completed
                    tracker.mark_completed(username, str(profile_file))
                    results["profiles_success"] += 1
                else:
                    logger.error(f"Failed to extract profile data for @{username}")
                    tracker.mark_error(username, "Failed to extract profile data")
                    results["profiles_failed"] += 1
                    results["errors"].append({
                        "username": username,
                        "error": "Failed to extract profile data"
                    })
            
            except Exception as e:
                logger.error(f"Error scraping @{username}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                
                tracker.mark_error(username, str(e))
                results["profiles_failed"] += 1
                results["errors"].append({
                    "username": username,
                    "error": str(e)
                })
            
            results["profiles_processed"] += 1
            
            # Progress update
            stats = tracker.get_stats()
            logger.info(f"\nProgress: {stats['completed']}/{stats['total']} completed, "
                       f"{stats['error']} errors, {stats['pending']} pending")
            
            # Delay between profiles
            if i < len(account_ids):
                delay = PROFILE_DELAY + random.uniform(0, 1)
                logger.info(f"Waiting {delay:.1f}s before next profile...")
                await asyncio.sleep(delay)
        
    finally:
        await scraper.stop()
        logger.info("Browser closed")
    
    # Final summary
    results["completed_at"] = datetime.now().isoformat()
    results["tracker_stats"] = tracker.get_stats()
    
    # Save run summary
    summary_file = output_dir / f"run_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info("SCRAPING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Profiles processed: {results['profiles_processed']}")
    logger.info(f"Success: {results['profiles_success']}")
    logger.info(f"Failed: {results['profiles_failed']}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"{'='*70}\n")
    
    tracker.close()
    return results


# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TikTok Profile Data Scraper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python crawl_profile.py
    python crawl_profile.py --list list32
    python crawl_profile.py --no-resume
    python crawl_profile.py --test
        """
    )
    
    parser.add_argument(
        '--list',
        type=str,
        default='list32',
        help='Account list file name (without .txt extension). Default: list32'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resume mode (rescrape all profiles)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: only scrape first 2 profiles'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run browser in headless mode'
    )
    
    parser.add_argument(
        '--use-xvfb',
        action='store_true',
        help='Use Xvfb for pseudo-headless mode (Linux only)'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()
    
    # Load account list
    list_file = Path(ACCOUNT_LIST_DIR) / f"{args.list}.txt"
    account_ids = load_account_ids(list_file)
    
    if not account_ids:
        print(f"Error: No accounts found in {list_file}")
        return 1
    
    print(f"Loaded {len(account_ids)} accounts from {list_file}")
    
    # Test mode: limit to 2 profiles
    if args.test:
        print("TEST MODE: Limiting to 2 profiles")
        account_ids = account_ids[:2]
    
    # Override headless setting
    global HEADLESS, USE_XVFB
    if args.headless:
        HEADLESS = True
    if args.use_xvfb:
        USE_XVFB = True
    
    # Setup logging
    logger = setup_logging(f"profile_{args.list}", verbose=args.verbose)
    
    # Run with or without Xvfb
    if USE_XVFB:
        logger.info("Using Xvfb pseudo-headless mode")
        with XvfbDisplay():
            await scrape_profile_data(
                account_ids=account_ids,
                resume=not args.no_resume,
                verbose=args.verbose,
                logger=logger,
                list_name=args.list
            )
    else:
        await scrape_profile_data(
            account_ids=account_ids,
            resume=not args.no_resume,
            verbose=args.verbose,
            logger=logger,
            list_name=args.list
        )
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
