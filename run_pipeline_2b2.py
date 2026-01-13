"""
Pipeline 2b2 Runner: Process URL list to fetch metadata + comments (resume mode)
- Reads URLs from video_list/YYYYMMDD/all_videos.txt
- Filters out usernames that already exist in comments_data/20260108_151848
- For each URL from uncompleted users: captures metadata via API + scrapes comments
- Saves to comments_data/YYYYMMDD/<username>/<video_id>.json
"""

import asyncio
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, List
import re

from TT_Content_Scraper.src.scraper_functions.playwright_scraper import PlaywrightScraper

DEFAULT_MAX_COMMENTS = 40
DEFAULT_RESTART_EVERY = 10
SLOW_MO = 50
DEFAULT_COMPLETED_FOLDERS = ["comments_data/20260108_151848", "comments_data/20260112_182216"]


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('playwright').setLevel(logging.WARNING)


def get_completed_usernames(completed_folders: list) -> Set[str]:
    """Get list of usernames that have already been scraped from multiple folders"""
    completed_users = set()
    for completed_folder in completed_folders:
        if os.path.exists(completed_folder):
            for item in os.listdir(completed_folder):
                item_path = os.path.join(completed_folder, item)
                if os.path.isdir(item_path):
                    completed_users.add(item)
    return completed_users


def extract_username_from_url(url: str) -> Optional[str]:
    """Extract username from TikTok URL"""
    match = re.search(r'@([^/]+)/', url)
    return match.group(1) if match else None


def filter_urls_by_uncompleted_users(
    input_file: str,
    completed_users: Set[str],
    logger
) -> List[str]:
    """Read URLs and filter out those from completed users"""
    uncompleted_urls = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            url = line.strip()
            if not url:
                continue
            
            username = extract_username_from_url(url)
            if username and username not in completed_users:
                uncompleted_urls.append(url)
    
    return uncompleted_urls


async def run_2b2(
    file_path: str,
    max_comments: int,
    headless: bool,
    restart_every: int,
    proxy: Optional[str],
    memory_restart_mb: Optional[int],
    completed_folders: list
) -> None:
    logger = logging.getLogger("Pipeline2b2")
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"Run timestamp: {run_timestamp}")
    
    # Get completed usernames
    logger.info(f"Checking completed folders: {completed_folders}")
    completed_users = get_completed_usernames(completed_folders)
    logger.info(f"Found {len(completed_users)} completed usernames: {sorted(completed_users)}")
    
    # Filter URLs
    logger.info(f"Reading URLs from {file_path}")
    uncompleted_urls = filter_urls_by_uncompleted_users(file_path, completed_users, logger)
    
    if not uncompleted_urls:
        logger.info("No URLs to process. All users have been completed!")
        return
    
    # Get unique usernames from uncompleted URLs
    unique_usernames = set()
    for url in uncompleted_urls:
        username = extract_username_from_url(url)
        if username:
            unique_usernames.add(username)
    
    logger.info(f"Total URLs to process: {len(uncompleted_urls)}")
    logger.info(f"Unique usernames to process: {len(unique_usernames)}")
    logger.info(f"Usernames: {sorted(unique_usernames)}")
    
    # Create temporary file with filtered URLs
    temp_file = f"temp_uncompleted_urls_{run_timestamp}.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for url in uncompleted_urls:
            f.write(f"{url}\n")
    logger.info(f"Created temporary file: {temp_file}")

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

    try:
        await scraper.start()
        logger.info("=" * 70)
        logger.info(f"2b2: Processing uncompleted URLs")
        logger.info(f"Max comments per video: {max_comments}")
        logger.info("=" * 70)

        # Override max_comments
        scraper._max_comments_override = max_comments
        
        await scraper.run_pipeline_2b_process_from_file(temp_file, run_timestamp)

    finally:
        await scraper.stop()
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logger.info(f"Removed temporary file: {temp_file}")


def main():
    parser = argparse.ArgumentParser(description="Run Pipeline 2b2 (Resume mode - skip completed users)")
    parser.add_argument("--file", type=str, required=True, help="Path to URL list file (e.g., video_list/20260108/all_videos.txt)")
    parser.add_argument("--completed-folders", type=str, nargs='+', default=DEFAULT_COMPLETED_FOLDERS, help="Paths to folders with completed users (can specify multiple)")
    parser.add_argument("--max-comments", type=int, default=DEFAULT_MAX_COMMENTS, help="Max comments per video")
    parser.add_argument("--restart-every", type=int, default=DEFAULT_RESTART_EVERY, help="Restart browser every N videos")
    parser.add_argument("--mem-restart-mb", type=int, default=None, help="Restart browser if RSS exceeds this MB")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--proxy", type=str, default=None, help="Proxy URL")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(args.verbose)

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return

    # Windows event loop policy
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(
        run_2b2(
            file_path=args.file,
            max_comments=args.max_comments,
            headless=args.headless,
            restart_every=args.restart_every,
            proxy=args.proxy,
            memory_restart_mb=args.mem_restart_mb,
            completed_folders=args.completed_folders,
        )
    )


if __name__ == "__main__":
    main()
