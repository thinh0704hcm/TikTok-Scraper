"""
Pipeline 2a Runner: URL list via API interception (network-only)
- Reads account IDs from crawl_account/<list>.txt or a provided file
- For each account, saves video URLs to video_list/YYYYMMDD/<username>.txt
"""

import asyncio
import argparse
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

from TT_Content_Scraper.src.scraper_functions.playwright_scraper import PlaywrightScraper

ACCOUNT_LIST_DIR = "crawl_account/"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_RESTART_EVERY = 5
SLOW_MO = 50


def load_account_ids(file_path: str) -> List[str]:
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


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('playwright').setLevel(logging.WARNING)


async def run_2a(
    account_ids: List[str],
    lookback_days: int,
    headless: bool,
    restart_every: int,
    proxy: Optional[str]
) -> None:
    logger = logging.getLogger("Pipeline2a")

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
        for i, username in enumerate(account_ids, start=1):
            logger.info("=" * 70)
            logger.info(f"2a: @{username} ({i}/{len(account_ids)})")
            logger.info("=" * 70)

            try:
                await scraper.run_pipeline_2a_fetch_list(username=username, lookback_days=lookback_days)
            except Exception as e:
                logger.error(f"Error on @{username}: {e}")

            # Restart between profiles as needed
            if i % restart_every == 0 and i != len(account_ids):
                await scraper.restart_browser()

    finally:
        await scraper.stop()


def main():
    parser = argparse.ArgumentParser(description="Run Pipeline 2a (URL list via API interception)")
    parser.add_argument("--list", type=str, default="list1", help="List name in crawl_account/")
    parser.add_argument("--account-file", type=str, default=None, help="Path to account list file")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Days to look back")
    parser.add_argument("--max-profiles", type=int, default=None, help="Limit profiles")
    parser.add_argument("--restart-every", type=int, default=DEFAULT_RESTART_EVERY, help="Restart browser every N profiles")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--proxy", type=str, default=None, help="Proxy URL")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.account_file:
        account_file = args.account_file
    else:
        account_file = str(Path(ACCOUNT_LIST_DIR) / f"{args.list}.txt")

    accounts = load_account_ids(account_file)
    if args.max_profiles:
        accounts = accounts[: args.max_profiles]

    if not accounts:
        print("No accounts to process.")
        return

    # Windows event loop policy
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(
        run_2a(
            account_ids=accounts,
            lookback_days=args.lookback,
            headless=args.headless,
            restart_every=args.restart_every,
            proxy=args.proxy,
        )
    )


if __name__ == "__main__":
    main()
