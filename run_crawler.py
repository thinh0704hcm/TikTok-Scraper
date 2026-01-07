"""
TikTok Profile Scraper Runner - UNIFIED DUAL-PIPELINE VERSION

✅ Pipeline 1: Fast metadata scraping (videos only)
✅ Pipeline 2: Deep scraping (videos + comments + labels)
✅ Memory-safe with browser restart strategy
✅ Memory monitoring for both pipelines
✅ Simple management - no database tracking
"""

import asyncio
import json
import logging
import sys
import os
import subprocess
import time
import random
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import unified scraper with both pipelines
try:
    from TT_Content_Scraper.src.scraper_functions.playwright_scraper import PlaywrightScraper
except ImportError:
    print("ERROR: Cannot import PlaywrightScraper. Make sure the unified version is available.")
    sys.exit(1)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed. Memory monitoring disabled.")
    print("Install with: pip install psutil")

# ============================================================================
# CONFIGURATION
# ============================================================================

ACCOUNT_LIST_DIR = "crawl_account/"
OUTPUT_BASE_DIR_P1 = "video_data/"          # Pipeline 1 output
OUTPUT_BASE_DIR_P2 = "comments_data/"       # Pipeline 2 output
MILESTONE_FILE = "milestone_datetime.txt"

DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_MAX_VIDEOS = 1000
DEFAULT_MAX_COMMENTS = 500

# Pipeline-specific settings
P1_PROFILE_DELAY_MIN = 5.0
P1_PROFILE_DELAY_MAX = 10.0
P1_RESTART_EVERY = 10

P2_PROFILE_DELAY_MIN = 10.0
P2_PROFILE_DELAY_MAX = 20.0
P2_RESTART_EVERY = 5  # More aggressive for Pipeline 2

SLOW_MO = 50
PROXY = None
HEADLESS = True

# ============================================================================
# MEMORY MONITOR
# ============================================================================

class MemoryMonitor:
    """Monitor and log memory usage"""
    
    def __init__(self):
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        self.baseline_memory = None
        self.profile_memory_log = []
        self.logger = logging.getLogger('MemoryMonitor')
    
    def get_memory_mb(self) -> float:
        if self.process:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0
    
    def set_baseline(self):
        if self.process:
            self.baseline_memory = self.get_memory_mb()
            self.logger.info(f"📊 Baseline memory: {self.baseline_memory:.1f} MB")
    
    def log_memory(self, label: str = "Current"):
        if not self.process:
            return None
        
        current = self.get_memory_mb()
        if self.baseline_memory:
            delta = current - self.baseline_memory
            self.logger.info(f"📊 {label}: {current:.1f} MB (Δ{delta:+.1f} MB)")
        else:
            self.logger.info(f"📊 {label}: {current:.1f} MB")
        return current
    
    def log_profile_memory(self, profile_num: int, username: str, before: float, after: float):
        if not self.process:
            return
        
        delta = after - before
        entry = {
            "profile_num": profile_num,
            "username": username,
            "memory_before_mb": round(before, 1),
            "memory_after_mb": round(after, 1),
            "memory_delta_mb": round(delta, 1),
            "timestamp": datetime.now().isoformat()
        }
        self.profile_memory_log.append(entry)
        
        self.logger.info(f"📊 Profile #{profile_num} (@{username}): {before:.1f} → {after:.1f} MB (Δ{delta:+.1f} MB)")
        
        if delta > 150:  # Higher threshold for Pipeline 2
            self.logger.warning(f"⚠️ Large memory increase: +{delta:.1f} MB")
    
    def save_memory_log(self, filepath: Path):
        if not self.profile_memory_log:
            return
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "baseline_mb": self.baseline_memory,
                    "profiles": self.profile_memory_log
                }, f, indent=2)
            self.logger.info(f"Memory log saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save memory log: {e}")

# ============================================================================
# DISPLAY MANAGEMENT
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
        if self.display and not self.use_xvfb:
            os.environ['DISPLAY'] = self.display
            self.logger.info(f"✓ Using existing display: {self.display}")
            return True
        
        if self.use_xvfb:
            return self._start_xvfb()
        
        return True
    
    def _start_xvfb(self) -> bool:
        try:
            subprocess.run(['which', 'Xvfb'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("Xvfb not installed. Install with: sudo apt-get install xvfb")
            return False
        
        try:
            self.process = subprocess.Popen(
                ['Xvfb', self.display, '-screen', '0', self.screen, '-ac', '-noreset'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            time.sleep(2)
            os.environ['DISPLAY'] = self.display
            self.logger.info(f"✓ Xvfb started on {self.display}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start Xvfb: {e}")
            return False
    
    def stop(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                pass
        
        if self.original_display:
            os.environ['DISPLAY'] = self.original_display

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_account_ids(file_path: str) -> List[str]:
    account_ids = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Warning: Account ID file not found: {file_path}")
        return account_ids
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            account_ids.append(line)
    
    return account_ids

def read_milestone_datetime(file_path: str = MILESTONE_FILE) -> Optional[datetime]:
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                return None
            
            try:
                return datetime.strptime(first_line, "%Y-%m-%d %H:%M:%S")
            except ValueError:
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
# PIPELINE 1: FAST METADATA SCRAPING
# ============================================================================

async def run_pipeline_1(
    account_ids: List[str],
    milestone_dt: Optional[datetime],
    max_videos: int = 1000,
    verbose: bool = False,
    headless: bool = True,
    proxy: Optional[str] = None,
    list_name: str = "list32",
    restart_browser_every: int = P1_RESTART_EVERY
) -> Dict[str, Any]:
    """
    Pipeline 1: Fast metadata scraping
    - Scrapes video metadata only
    - Outputs to video_data/
    """
    
    logger = setup_logging(f"pipeline1_{list_name}", verbose=verbose)
    
    logger.info("=" * 70)
    logger.info("PIPELINE 1: FAST METADATA SCRAPING")
    logger.info("=" * 70)
    logger.info(f"Target Profiles: {len(account_ids)}")
    if milestone_dt:
        logger.info(f"Milestone Date: {milestone_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        logger.info("Milestone: None (Scraping all videos)")
    logger.info(f"Browser Restart: Every {restart_browser_every} profiles")
    logger.info("=" * 70)
    
    # Initialize Memory Monitor
    memory_monitor = MemoryMonitor()
    memory_monitor.set_baseline()
    
    if not account_ids:
        logger.info("No profiles to scrape.")
        return {}

    # Setup Output Directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(OUTPUT_BASE_DIR_P1) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {run_dir}")
    
    # Track run statistics
    run_stats = {
        "pipeline": 1,
        "list_name": list_name,
        "milestone_datetime": milestone_dt.strftime('%Y-%m-%d %H:%M:%S') if milestone_dt else None,
        "output_dir": str(run_dir),
        "started_at": datetime.now().isoformat(),
        "profiles_processed": 0,
        "profiles_success": 0,
        "profiles_failed": 0,
        "total_videos": 0,
        "profile_summaries": {}
    }

    # Initialize scraper
    scraper = PlaywrightScraper(
        headless=headless,
        slow_mo=SLOW_MO,
        wait_time=5.0,
        proxy=proxy,
        fingerprint_file='browser_fingerprint.json',
        restart_browser_every=restart_browser_every,
        max_console_logs=1000
    )
    
    try:
        await scraper.start()
        memory_monitor.log_memory("After browser start")
        
        for i, account_id in enumerate(account_ids):
            profile_num = i + 1
            
            # Restart browser periodically
            if profile_num > 1 and profile_num % restart_browser_every == 0:
                logger.info("=" * 70)
                logger.info(f"🔄 Processed {profile_num - 1} profiles")
                logger.info("🔄 Restarting browser...")
                logger.info("=" * 70)
                
                mem_before_restart = memory_monitor.get_memory_mb()
                await scraper.restart_browser()
                mem_after_restart = memory_monitor.get_memory_mb()
                logger.info(f"📊 Memory freed: {mem_before_restart - mem_after_restart:.1f} MB")
                
                await asyncio.sleep(5)
            
            try:
                logger.info("=" * 70)
                logger.info(f"PROFILE {profile_num}/{len(account_ids)}: @{account_id}")
                logger.info("=" * 70)
                
                mem_before = memory_monitor.get_memory_mb()
                run_stats["profiles_processed"] += 1
                
                profile_dir = run_dir / account_id
                
                summary = await scraper.scrape_user_profile(
                    username=account_id,
                    profile_dir=profile_dir,
                    max_videos=max_videos,
                    milestone_datetime=milestone_dt
                )
                
                mem_after = memory_monitor.get_memory_mb()
                memory_monitor.log_profile_memory(profile_num, account_id, mem_before, mem_after)
                
                if "error" in summary:
                    logger.warning(f"❌ Failed: {summary['error']}")
                    run_stats["profiles_failed"] += 1
                else:
                    videos = summary.get('total_videos', 0)
                    logger.info(f"✅ Success: {videos} videos")
                    run_stats["profiles_success"] += 1
                    run_stats["total_videos"] += videos
                    
                    run_stats["profile_summaries"][account_id] = {
                        "username": account_id,
                        "videos": videos,
                        "date_range": summary.get('date_range'),
                        "files": summary.get('files'),
                        "memory_delta_mb": round(mem_after - mem_before, 1)
                    }
                
                if account_id != account_ids[-1]:
                    sleep_time = random.uniform(P1_PROFILE_DELAY_MIN, P1_PROFILE_DELAY_MAX)
                    logger.info(f"💤 Sleeping {sleep_time:.1f}s...")
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                run_stats["profiles_failed"] += 1
                
                try:
                    logger.info("Attempting recovery...")
                    await scraper.restart_browser()
                    await asyncio.sleep(10)
                except Exception as recovery_error:
                    logger.error(f"Recovery failed: {recovery_error}")
    
    finally:
        logger.info("=" * 70)
        logger.info("CLEANING UP")
        logger.info("=" * 70)
        
        memory_monitor.log_memory("Before cleanup")
        await scraper.stop()
        gc.collect()
        memory_monitor.log_memory("After cleanup")
    
    # Save run summary
    run_stats["completed_at"] = datetime.now().isoformat()
    run_stats["scraper_stats"] = scraper.stats
    
    run_summary_file = run_dir / f"run_summary_{timestamp}.json"
    with open(run_summary_file, 'w', encoding='utf-8') as f:
        json.dump(run_stats, f, ensure_ascii=False, indent=2)
    
    memory_log_file = run_dir / f"memory_log_{timestamp}.json"
    memory_monitor.save_memory_log(memory_log_file)
    
    logger.info("=" * 70)
    logger.info("PIPELINE 1 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"✅ Summary: {run_summary_file}")
    logger.info(f"📊 Memory log: {memory_log_file}")
    logger.info(f"   - Profiles: {run_stats['profiles_processed']}")
    logger.info(f"   - Success: {run_stats['profiles_success']}")
    logger.info(f"   - Failed: {run_stats['profiles_failed']}")
    logger.info(f"   - Total videos: {run_stats['total_videos']}")
    logger.info(f"   - Browser restarts: {scraper.stats['browser_restarts']}")
    logger.info("=" * 70)
    
    return run_stats

# ============================================================================
# PIPELINE 2: DEEP SCRAPING WITH COMMENTS
# ============================================================================

async def run_pipeline_2(
    account_ids: List[str],
    lookback_days: int,
    max_comments_per_video: int = 500,
    verbose: bool = False,
    headless: bool = True,
    proxy: Optional[str] = None,
    list_name: str = "list32",
    restart_browser_every: int = P2_RESTART_EVERY
) -> Dict[str, Any]:
    """
    Pipeline 2: Deep scraping with comments and labels
    - Scrapes videos + comments + diversification labels
    - No progress tracking (processes each video individually)
    - Outputs to comments_data/{date}/{username}/{video_id}.json
    """
    
    logger = setup_logging(f"pipeline2_{list_name}", verbose=verbose)
    
    logger.info("=" * 70)
    logger.info("PIPELINE 2: DEEP SCRAPING WITH COMMENTS")
    logger.info("=" * 70)
    logger.info(f"Target Profiles: {len(account_ids)}")
    logger.info(f"Lookback: {lookback_days} days")
    logger.info(f"Max Comments/Video: {max_comments_per_video}")
    logger.info(f"Browser Restart: Every {restart_browser_every} profiles")
    logger.info("=" * 70)
    
    # Initialize Memory Monitor
    memory_monitor = MemoryMonitor()
    memory_monitor.set_baseline()
    
    # Track run statistics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_stats = {
        "pipeline": 2,
        "list_name": list_name,
        "lookback_days": lookback_days,
        "max_comments_per_video": max_comments_per_video,
        "output_dir": str(Path(OUTPUT_BASE_DIR_P2)),
        "started_at": datetime.now().isoformat(),
        "profiles_processed": 0,
        "profiles_success": 0,
        "profiles_failed": 0,
        "total_videos": 0,
        "total_comments": 0,
        "profile_summaries": {}
    }
    
    # Initialize scraper
    scraper = PlaywrightScraper(
        headless=headless,
        slow_mo=SLOW_MO,
        wait_time=5.0,
        proxy=proxy,
        fingerprint_file='browser_fingerprint.json',
        restart_browser_every=restart_browser_every,
        max_console_logs=1000
    )
    
    try:
        await scraper.start()
        memory_monitor.log_memory("After browser start")
        
        for i, account_id in enumerate(account_ids):
            profile_num = i + 1
            
            # Restart browser between profiles (Pipeline 2 is more resource-intensive)
            if profile_num > 1:
                logger.info("=" * 70)
                logger.info(f"🔄 Restarting browser after profile {profile_num - 1}")
                logger.info("=" * 70)
                
                mem_before_restart = memory_monitor.get_memory_mb()
                await scraper.restart_browser()
                mem_after_restart = memory_monitor.get_memory_mb()
                logger.info(f"📊 Memory freed: {mem_before_restart - mem_after_restart:.1f} MB")
                
                await asyncio.sleep(5)
            
            try:
                logger.info("=" * 70)
                logger.info(f"PROFILE {profile_num}/{len(account_ids)}: @{account_id}")
                logger.info("=" * 70)
                
                mem_before = memory_monitor.get_memory_mb()
                run_stats["profiles_processed"] += 1
                
                # Run Pipeline 2 detailed scraping
                await scraper.run_pipeline_2_detailed(
                    username=account_id,
                    lookback_days=lookback_days,
                    max_comments_per_video=max_comments_per_video
                )
                
                mem_after = memory_monitor.get_memory_mb()
                memory_monitor.log_profile_memory(profile_num, account_id, mem_before, mem_after)
                
                # Count output files to track success
                date_str = datetime.now().strftime('%Y%m%d')
                output_dir = Path(OUTPUT_BASE_DIR_P2) / date_str / account_id
                
                if output_dir.exists():
                    video_files = list(output_dir.glob("*.json"))
                    total_comments = 0
                    
                    for video_file in video_files:
                        try:
                            with open(video_file, 'r') as f:
                                data = json.load(f)
                                total_comments += data.get('comments_count', 0)
                        except:
                            pass
                    
                    logger.info(f"✅ Success: {len(video_files)} videos, {total_comments} comments")
                    run_stats["profiles_success"] += 1
                    run_stats["total_videos"] += len(video_files)
                    run_stats["total_comments"] += total_comments
                    
                    run_stats["profile_summaries"][account_id] = {
                        "username": account_id,
                        "videos": len(video_files),
                        "comments": total_comments,
                        "output_dir": str(output_dir),
                        "memory_delta_mb": round(mem_after - mem_before, 1)
                    }
                else:
                    logger.warning(f"⚠️ No output directory found for @{account_id}")
                    run_stats["profiles_failed"] += 1
                
                if account_id != account_ids[-1]:
                    sleep_time = random.uniform(P2_PROFILE_DELAY_MIN, P2_PROFILE_DELAY_MAX)
                    logger.info(f"💤 Sleeping {sleep_time:.1f}s...")
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                run_stats["profiles_failed"] += 1
                
                try:
                    logger.info("Attempting recovery...")
                    await scraper.restart_browser()
                    await asyncio.sleep(10)
                except Exception as recovery_error:
                    logger.error(f"Recovery failed: {recovery_error}")
    
    finally:
        logger.info("=" * 70)
        logger.info("CLEANING UP")
        logger.info("=" * 70)
        
        memory_monitor.log_memory("Before cleanup")
        await scraper.stop()
        gc.collect()
        memory_monitor.log_memory("After cleanup")
    
    # Save run summary
    run_stats["completed_at"] = datetime.now().isoformat()
    run_stats["scraper_stats"] = scraper.stats
    
    # Save to output directory
    summary_dir = Path("logs/pipeline2_summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_summary_file = summary_dir / f"p2_summary_{timestamp}.json"
    
    with open(run_summary_file, 'w', encoding='utf-8') as f:
        json.dump(run_stats, f, ensure_ascii=False, indent=2)
    
    memory_log_file = summary_dir / f"p2_memory_{timestamp}.json"
    memory_monitor.save_memory_log(memory_log_file)
    
    logger.info("=" * 70)
    logger.info("PIPELINE 2 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"✅ Summary: {run_summary_file}")
    logger.info(f"📊 Memory log: {memory_log_file}")
    logger.info(f"   - Profiles: {run_stats['profiles_processed']}")
    logger.info(f"   - Success: {run_stats['profiles_success']}")
    logger.info(f"   - Failed: {run_stats['profiles_failed']}")
    logger.info(f"   - Total videos: {run_stats['total_videos']}")
    logger.info(f"   - Total comments: {run_stats['total_comments']}")
    logger.info(f"   - Browser restarts: {scraper.stats['browser_restarts']}")
    logger.info("=" * 70)
    
    return run_stats

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="TikTok Unified Scraper - Dual Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Selection:
  --pipeline 1    Fast metadata scraping (videos only)
  --pipeline 2    Deep scraping (videos + comments + labels)

Examples:
  # Pipeline 1: Fast scraping with 30-day lookback
  python run_crawler.py --pipeline 1 --list mylist --lookback 30

  # Pipeline 2: Deep scraping with comments
  python run_crawler.py --pipeline 2 --list mylist --lookback 7 --max-comments 300

  # Test mode (first 2 profiles only)
  python run_crawler.py --pipeline 1 --test
        """
    )
    
    # Pipeline Selection
    parser.add_argument("--pipeline", type=int, choices=[1, 2], required=True,
                        help="Pipeline: 1=Fast metadata, 2=Deep with comments")
    
    # Input/Output
    parser.add_argument("--list", type=str, default="list32", help="Name of list in crawl_account/")
    parser.add_argument("--account-file", type=str, default=None, help="Direct path to account file")
    
    # Limits & Logic
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS, 
                        help="Days to look back")
    parser.add_argument("--max-videos", type=int, default=DEFAULT_MAX_VIDEOS,
                        help="[P1] Max videos per profile")
    parser.add_argument("--max-comments", type=int, default=DEFAULT_MAX_COMMENTS,
                        help="[P2] Max comments per video")
    parser.add_argument("--max-profiles", type=int, default=None,
                        help="Limit number of profiles to process")
    parser.add_argument("--test", action="store_true", 
                        help="Test mode: first 2 profiles only")
    parser.add_argument("--restart-every", type=int, default=None, 
                        help="Restart browser every N profiles (default: P1=10, P2=5)")
    
    # Technical
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--xvfb", action="store_true", help="Use Xvfb for display")
    parser.add_argument("--display", type=str, default=None, help="Display to use (e.g. :2)")
    parser.add_argument("--proxy", type=str, default=PROXY, help="Proxy URL")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()

    # Display Management
    display_to_use = args.display
    use_xvfb = args.xvfb
    if not display_to_use and not use_xvfb and sys.platform == 'linux' and not os.environ.get('DISPLAY'):
        use_xvfb = True
        
    headless_mode = args.headless or HEADLESS
    if display_to_use: 
        headless_mode = False

    # Load Account List
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
        print(f"🧪 Test mode: Running on first 2 profiles")
    elif args.max_profiles: 
        account_ids = account_ids[:args.max_profiles]

    # Set restart frequency
    if args.restart_every:
        restart_every = args.restart_every
    else:
        restart_every = P1_RESTART_EVERY if args.pipeline == 1 else P2_RESTART_EVERY

    # Run Selected Pipeline
    with DisplayManager(display=display_to_use, use_xvfb=use_xvfb) as disp:
        if args.pipeline == 1:
            # Determine Milestone for Pipeline 1
            milestone_dt = read_milestone_datetime()
            if not milestone_dt and args.lookback > 0:
                milestone_dt = datetime.now() - timedelta(days=args.lookback)
                print(f"📅 Milestone: {milestone_dt.strftime('%Y-%m-%d')} ({args.lookback} days ago)")
            elif milestone_dt:
                print(f"📅 Milestone from file: {milestone_dt.strftime('%Y-%m-%d')}")
            
            asyncio.run(run_pipeline_1(
                account_ids=account_ids,
                milestone_dt=milestone_dt,
                max_videos=args.max_videos,
                verbose=args.verbose,
                headless=headless_mode,
                proxy=args.proxy,
                list_name=list_name,
                restart_browser_every=restart_every
            ))
        
        elif args.pipeline == 2:
            print(f"📅 Lookback: {args.lookback} days")
            print(f"💬 Max comments per video: {args.max_comments}")
            
            asyncio.run(run_pipeline_2(
                account_ids=account_ids,
                lookback_days=args.lookback,
                max_comments_per_video=args.max_comments,
                verbose=args.verbose,
                headless=headless_mode,
                proxy=args.proxy,
                list_name=list_name,
                restart_browser_every=restart_every
            ))

if __name__ == "__main__":
    main()