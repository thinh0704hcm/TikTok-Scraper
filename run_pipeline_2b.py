"""
Pipeline 2b Runner: Process URL list to fetch metadata + comments (network-only)
- Reads URLs from video_list/YYYYMMDD/<username>.txt
- For each URL: captures metadata via API + scrapes comments
- Saves to comments_data/YYYYMMDD/<username>/<video_id>.json
"""

import asyncio
import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from TT_Content_Scraper.src.scraper_functions.playwright_scraper import PlaywrightScraper

DEFAULT_MAX_COMMENTS = 500
DEFAULT_RESTART_EVERY = 5
SLOW_MO = 50


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('playwright').setLevel(logging.WARNING)


async def run_2b(
    file_path: str,
    max_comments: int,
    headless: bool,
    restart_every: int,
    proxy: Optional[str]
) -> None:
    logger = logging.getLogger("Pipeline2b")

    scraper = PlaywrightScraper(
        headless=headless,
        slow_mo=SLOW_MO,
        wait_time=5.0,
        proxy=proxy,
        fingerprint_file='browser_fingerprint.json',
        restart_browser_every=restart_every,
        max_console_logs=1000,
    )

    try:
        await scraper.start()
        logger.info("=" * 70)
        logger.info(f"2b: Processing {file_path}")
        logger.info(f"Max comments per video: {max_comments}")
        logger.info("=" * 70)

        # Override max_comments in get_video_comments calls
        # We'll pass it via the method call in run_pipeline_2b_process_from_file
        scraper._max_comments_override = max_comments
        
        await scraper.run_pipeline_2b_process_from_file(file_path)

    finally:
        await scraper.stop()


def main():
    parser = argparse.ArgumentParser(description="Run Pipeline 2b (Process URL list → metadata + comments)")
    parser.add_argument("--file", type=str, required=True, help="Path to URL list file (e.g., video_list/20260107/username.txt)")
    parser.add_argument("--max-comments", type=int, default=DEFAULT_MAX_COMMENTS, help="Max comments per video")
    parser.add_argument("--restart-every", type=int, default=DEFAULT_RESTART_EVERY, help="Restart browser every N videos")
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
        run_2b(
            file_path=args.file,
            max_comments=args.max_comments,
            headless=args.headless,
            restart_every=args.restart_every,
            proxy=args.proxy,
        )
    )


if __name__ == "__main__":
    main()
