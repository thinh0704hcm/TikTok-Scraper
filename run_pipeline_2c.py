"""
Pipeline 2c Runner: Process URL list to fetch metadata (network-only)
- Reads URLs from video_list/YYYYMMDD/<username>.txt
- For each URL: captures metadata via API
"""

import asyncio
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from TT_Content_Scraper.src.scraper_functions.playwright_scraper import PlaywrightScraper

DEFAULT_RESTART_EVERY = 20  # Higher default to avoid frequent restarts; tune manually
SLOW_MO = 50  # Lower slow-mo to speed up interactions
PROXY = "118.70.171.121:53347:thinh:thinh"

def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('playwright').setLevel(logging.WARNING)


async def run_2c(
    file_path: str,
    headless: bool,
    restart_every: int,
    proxy: Optional[str],
    memory_restart_mb: Optional[int]
) -> None:
    logger = logging.getLogger("Pipeline2c")
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"Run timestamp: {run_timestamp}")

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
        logger.info(f"2c: Processing {file_path}")
        logger.info("=" * 70)

        await scraper.run_pipeline_2c_process_from_file(file_path, run_timestamp)

    finally:
        await scraper.stop()


def main():
    parser = argparse.ArgumentParser(description="Run Pipeline 2c (Process URL list → metadata)")
    parser.add_argument("--file", type=str, required=True, help="Path to URL list file (e.g., video_list/20260107/username.txt)")
    parser.add_argument("--restart-every", type=int, default=DEFAULT_RESTART_EVERY, help="Restart browser every N videos")
    parser.add_argument("--mem-restart-mb", type=int, default=5000, help="Restart browser if RSS exceeds this MB")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--proxy", type=str, default=PROXY, help="Proxy URL")
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
        run_2c(
            file_path=args.file,
            headless=args.headless,
            restart_every=args.restart_every,
            proxy=args.proxy,
            memory_restart_mb=args.mem_restart_mb,
        )
    )


if __name__ == "__main__":
    main()
