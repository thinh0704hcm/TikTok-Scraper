"""
Pipeline 4 Runner: Bulk Thumbnail Downloading
- Reads account IDs from crawl_account/<list>.txt or a provided file
- For each account, downloads thumbnails to thumbnails/<timestamp>/<username>/
- Saves metadata with local thumbnail paths
"""

import asyncio
import argparse
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from TT_Content_Scraper.src.scraper_functions.playwright_scraper import PlaywrightScraper

ACCOUNT_LIST_DIR = "crawl_account/"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_RESTART_EVERY = 10
DEFAULT_QUALITY = '960'
SLOW_MO = 50
PROXY = "118.70.171.121:53347:thinh:thinh"


def load_account_ids(file_path: str) -> List[str]:
    """Load account IDs from text file"""
    account_ids: List[str] = []
    p = Path(file_path)
    if not p.exists():
        print(f"Warning: Account file not found: {file_path}")
        return account_ids
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('//'):
                continue
            account_ids.append(s)
    return account_ids


def setup_logging(verbose: bool, log_file: Optional[str] = None) -> None:
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logging.getLogger('playwright').setLevel(logging.WARNING)


async def run_4(
    account_ids: List[str],
    lookback_days: int,
    quality: str,
    headless: bool,
    restart_every: int,
    proxy: Optional[str],
    memory_restart_mb: Optional[int],
    skip_existing: bool
) -> None:
    """
    Run Pipeline 4: Bulk thumbnail downloading
    
    Args:
        account_ids: List of TikTok usernames
        lookback_days: Only fetch videos from last N days
        quality: Thumbnail quality ('960', '720', '480', '240', 'origin')
        headless: Run browser in headless mode
        restart_every: Restart browser every N profiles
        proxy: Proxy URL
        memory_restart_mb: Restart browser if memory exceeds this threshold
        skip_existing: Skip users that already have thumbnails downloaded
    """
    logger = logging.getLogger("Pipeline4")
    
    # Create single run timestamp for all profiles
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info("=" * 70)
    logger.info("PIPELINE 4: BULK THUMBNAIL DOWNLOADING")
    logger.info("=" * 70)
    logger.info(f"Run timestamp: {run_timestamp}")
    logger.info(f"Total accounts: {len(account_ids)}")
    logger.info(f"Lookback days: {lookback_days}")
    logger.info(f"Thumbnail quality: {quality}p")
    logger.info(f"Restart every: {restart_every} profiles")
    logger.info(f"Memory restart: {memory_restart_mb} MB" if memory_restart_mb else "Memory restart: Disabled")
    logger.info("=" * 70)

    scraper = PlaywrightScraper(
        headless=headless,
        slow_mo=SLOW_MO,
        wait_time=5.0,
        proxy=proxy,
        fingerprint_file='browser_fingerprint.json',
        restart_browser_every=restart_every,
        max_console_logs=1000,
        memory_restart_mb=memory_restart_mb,
    )

    # Track statistics
    stats = {
        'total_accounts': len(account_ids),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'total_videos': 0,
        'total_downloaded': 0,
        'total_failed': 0
    }

    try:
        await scraper.start()
        
        for i, username in enumerate(account_ids, start=1):
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"ACCOUNT {i}/{len(account_ids)}: @{username}")
            logger.info("=" * 70)

            # Check if already processed (skip_existing)
            if skip_existing:
                user_dir = Path("thumbnails") / run_timestamp / username
                if user_dir.exists() and any(user_dir.glob("*.jpg")):
                    logger.info(f"⏭️  Skipping @{username} (already has thumbnails)")
                    stats['skipped'] += 1
                    continue

            try:
                result = await scraper.run_pipeline_4_thumbnails(
                    usernames=[username],
                    lookback_days=lookback_days,
                    run_timestamp=run_timestamp,
                    quality=quality
                )
                
                # Update statistics
                stats['processed'] += 1
                if result:
                    stats['total_videos'] += result.get('total_videos', 0)
                    stats['total_downloaded'] += result.get('total_downloaded', 0)
                    stats['total_failed'] += result.get('total_failed', 0)
                
                logger.info(f"✅ Completed @{username}")
                
            except Exception as e:
                logger.error(f"❌ Error on @{username}: {e}", exc_info=True)
                stats['failed'] += 1
                continue

            # Restart browser periodically
            if i % restart_every == 0 and i != len(account_ids):
                logger.info("♻️  Restarting browser...")
                await scraper.restart_browser()

        # Final summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE 4 COMPLETE - FINAL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total accounts: {stats['total_accounts']}")
        logger.info(f"Processed: {stats['processed']}")
        logger.info(f"Skipped: {stats['skipped']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Total videos found: {stats['total_videos']}")
        logger.info(f"Total thumbnails downloaded: {stats['total_downloaded']}")
        logger.info(f"Total download failures: {stats['total_failed']}")
        if stats['total_videos'] > 0:
            success_rate = (stats['total_downloaded'] / stats['total_videos']) * 100
            logger.info(f"Download success rate: {success_rate:.1f}%")
        logger.info(f"Output directory: thumbnails/{run_timestamp}/")
        logger.info("=" * 70)
        
        # Save final summary file
        summary_file = Path("thumbnails") / run_timestamp / f"_summary_{run_timestamp}.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'pipeline': 4,
                'run_timestamp': run_timestamp,
                'quality': quality,
                'lookback_days': lookback_days,
                'statistics': stats,
                'completed_at': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to: {summary_file}")

    finally:
        await scraper.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Run Pipeline 4: Bulk Thumbnail Downloading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 960p thumbnails for accounts in list1.txt (last 30 days)
  python run_pipeline_4.py --list list1
  
  # Download 720p thumbnails with 7-day lookback
  python run_pipeline_4.py --list list1 --quality 720 --lookback 7
  
  # Use custom account file
  python run_pipeline_4.py --account-file my_accounts.txt
  
  # Run headless with proxy
  python run_pipeline_4.py --list list1 --headless --proxy http://user:pass@proxy:8080
  
  # Limit to first 10 accounts
  python run_pipeline_4.py --list list1 --max-profiles 10
  
  # Restart browser every 5 profiles or if memory exceeds 2GB
  python run_pipeline_4.py --list list1 --restart-every 5 --mem-restart-mb 2000
        """
    )
    
    # Input options
    parser.add_argument(
        "--list", 
        type=str, 
        default="list1", 
        help="List name in crawl_account/ directory (default: list1)"
    )
    parser.add_argument(
        "--account-file", 
        type=str, 
        default=None, 
        help="Path to account list file (overrides --list)"
    )
    
    # Scraping options
    parser.add_argument(
        "--lookback", 
        type=int, 
        default=DEFAULT_LOOKBACK_DAYS, 
        help=f"Days to look back for videos (default: {DEFAULT_LOOKBACK_DAYS})"
    )
    parser.add_argument(
        "--quality", 
        type=str, 
        default=DEFAULT_QUALITY,
        choices=['960', '720', '480', '240', 'origin'],
        help=f"Thumbnail quality (default: {DEFAULT_QUALITY})"
    )
    parser.add_argument(
        "--max-profiles", 
        type=int, 
        default=None, 
        help="Limit number of profiles to process"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip accounts that already have thumbnails downloaded"
    )
    
    # Browser options
    parser.add_argument(
        "--restart-every", 
        type=int, 
        default=DEFAULT_RESTART_EVERY, 
        help=f"Restart browser every N profiles (default: {DEFAULT_RESTART_EVERY})"
    )
    parser.add_argument(
        "--mem-restart-mb", 
        type=int, 
        default=None, 
        help="Restart browser if RSS memory exceeds this threshold (MB)"
    )
    parser.add_argument(
        "--headless", 
        action="store_true", 
        help="Run browser in headless mode"
    )
    parser.add_argument(
        "--proxy", 
        type=str, 
        default=PROXY, 
        help="Proxy URL (format: http://user:pass@host:port or host:port:user:pass)"
    )
    
    # Logging options
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose (DEBUG level) logging"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Save logs to file (default: logs/pipeline4_TIMESTAMP.log)"
    )

    args = parser.parse_args()

    # Setup logging
    if args.log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        args.log_file = str(log_dir / f"pipeline4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    setup_logging(args.verbose, args.log_file)
    logger = logging.getLogger("Pipeline4Runner")

    # Load account list
    if args.account_file:
        account_file = args.account_file
    else:
        account_file = str(Path(ACCOUNT_LIST_DIR) / f"{args.list}.txt")

    accounts = load_account_ids(account_file)
    
    if args.max_profiles:
        accounts = accounts[:args.max_profiles]
        logger.info(f"Limited to first {args.max_profiles} accounts")

    if not accounts:
        logger.error("No accounts to process. Exiting.")
        return

    logger.info(f"Loaded {len(accounts)} accounts from {account_file}")

    # Windows event loop policy
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Run pipeline
    try:
        asyncio.run(
            run_4(
                account_ids=accounts,
                lookback_days=args.lookback,
                quality=args.quality,
                headless=args.headless,
                restart_every=args.restart_every,
                proxy=args.proxy,
                memory_restart_mb=args.mem_restart_mb,
                skip_existing=args.skip_existing
            )
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()