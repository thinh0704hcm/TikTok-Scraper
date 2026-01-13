"""
Pipeline 3 Runner: Explore page scraping - repeatedly crawl for random videos
- Scrapes TikTok explore page multiple times
- Filters by language
- Collects consolidated unique videos
"""

import asyncio
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from TT_Content_Scraper.src.scraper_functions.playwright_scraper import PlaywrightScraper

DEFAULT_RESTART_EVERY = 15
SLOW_MO = 50
PROXY = "118.70.171.121:53347:thinh:thinh"

def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('playwright').setLevel(logging.WARNING)


async def run_pipeline_3(
    target_video_count: int,
    lookback_days: int,
    language: str,
    headless: bool,
    restart_every: int,
    proxy: Optional[str],
    memory_restart_mb: Optional[int]
) -> None:
    logger = logging.getLogger("Pipeline3")
    
    # Calculate runs and items per run
    num_runs = max(1, target_video_count // 100)
    max_items_per_run = max(100, target_video_count // num_runs)
    
    logger.info(f"Target video count: {target_video_count}")
    logger.info(f"Lookback days: {lookback_days}")
    logger.info(f"Calculated runs: {num_runs} × {max_items_per_run} items/run")

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
        logger.info(f"3: Explore page scraping")
        logger.info(f"   Target: {target_video_count} videos")
        logger.info(f"   Lookback: {lookback_days} days")
        logger.info(f"   Language: {language}")
        logger.info("=" * 70)

        summary = await scraper.run_pipeline_3_explore(
            num_runs=num_runs,
            max_items_per_run=max_items_per_run,
            lookback_days=lookback_days,
            lang=language,
            delay_between_runs=(5, 15)
        )
        
        logger.info(f"\n📊 Final Summary:")
        logger.info(f"   Total videos collected: {summary['total_videos_collected']}")
        logger.info(f"   Unique videos: {summary['unique_videos']}")
        logger.info(f"   Output: {summary['output_directory']}")

    finally:
        await scraper.stop()


def main():
    parser = argparse.ArgumentParser(description="Run Pipeline 3 (Explore scraping)")
    parser.add_argument(
        "--target",
        type=int,
        required=True,
        help="Target number of videos to collect"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=0,
        help="Limit to videos from last N days (0 = no limit)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="vi",
        help="Language to filter (default: vi for Vietnamese)"
    )
    parser.add_argument(
        "--restart-every",
        type=int,
        default=DEFAULT_RESTART_EVERY,
        help="Restart browser every N videos"
    )
    parser.add_argument(
        "--mem-restart-mb",
        type=int,
        default=5000,
        help="Restart browser if RSS exceeds this MB"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless"
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=PROXY,
        help="Proxy URL"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Windows event loop policy
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(
        run_pipeline_3(
            target_video_count=args.target,
            lookback_days=args.lookback_days,
            language=args.lang,
            headless=args.headless,
            restart_every=args.restart_every,
            proxy=args.proxy,
            memory_restart_mb=args.mem_restart_mb,
        )
    )


if __name__ == "__main__":
    main()
