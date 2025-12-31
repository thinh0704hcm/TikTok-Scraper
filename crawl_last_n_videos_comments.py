"""
TikTok Video Comments Crawler

Crawls all comments from the N most recent videos of a TikTok account.

Usage:
    python crawl_last_n_videos_comments.py bac_hello2              # Default: 3 videos
    python crawl_last_n_videos_comments.py bac_hello2 --n 5        # Crawl 5 videos
    python crawl_last_n_videos_comments.py bac_hello2 --headless   # Run in headless mode
"""

import asyncio
import json
import logging
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from TT_Content_Scraper.src.scraper_functions.playwright_scraper import (
    PlaywrightProfileScraper, VideoData
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Output settings
OUTPUT_BASE_DIR = "video_comments_data/"

# Browser settings
IS_HEADLESS_SERVER = os.environ.get('DISPLAY') is None and sys.platform.startswith('linux')
HEADLESS = IS_HEADLESS_SERVER
SLOW_MO = 50

# Hardcoded proxy (set your proxy here)
PROXY = "14.224.198.119:44182:HcgsFh:ZnHhhU"

# Default number of videos to crawl
DEFAULT_N_VIDEOS = 3


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CommentData:
    """Data class for comment information"""
    comment_id: str
    video_id: str
    author_username: str
    author_unique_id: str
    text: str
    create_time: int
    create_time_iso: str
    like_count: int
    reply_count: int
    is_author_verified: bool
    scraped_at: str


# ============================================================================
# COMMENT SCRAPER
# ============================================================================

class CommentScraper(PlaywrightProfileScraper):
    """Extended scraper for TikTok comments"""
    
    async def get_video_comments(
        self,
        video_id: str,
        max_comments: int = 10000
    ) -> List[CommentData]:
        """
        Get all comments from a specific video.
        
        Args:
            video_id: TikTok video ID
            max_comments: Maximum number of comments to fetch
        
        Returns:
            List of comment data
        """
        url = f"https://www.tiktok.com/@placeholder/video/{video_id}"
        
        comments = []
        seen_ids = set()
        
        try:
            # Clear previous responses
            self.clear_responses()
            
            logger.info(f"Navigating to video {video_id}...")
            
            # Try navigation with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    break
                except Exception as nav_err:
                    if attempt < max_retries - 1:
                        wait_time = 3 * (attempt + 1)
                        logger.warning(f"Navigation attempt {attempt+1} failed, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
            
            await asyncio.sleep(self.wait_time + 2)
            
            # Wait for comment section or video content
            try:
                await self.page.wait_for_selector('[data-e2e="browse-video"]', timeout=10000)
            except:
                logger.warning("Video content not found, continuing anyway...")
            
            # Try to click on comment section to expand it
            try:
                comment_button = await self.page.query_selector('[data-e2e="browse-comment"]')
                if comment_button:
                    await comment_button.click()
                    await asyncio.sleep(1)
            except Exception as e:
                logger.debug(f"Could not click comment button: {e}")
            
            # Scroll to load comments
            scroll_count = 0
            max_scrolls = 100
            no_new_comments_count = 0
            processed_urls = set()
            scrolls_without_response = 0
            
            while len(comments) < max_comments and scroll_count < max_scrolls:
                # Scroll in comment section or main page
                try:
                    # Try scrolling in comment container first
                    await self.page.evaluate('''
                        () => {
                            const commentContainer = document.querySelector('[data-e2e="comment-list"]');
                            if (commentContainer) {
                                commentContainer.scrollBy(0, commentContainer.clientHeight);
                            } else {
                                window.scrollBy(0, window.innerHeight);
                            }
                        }
                    ''')
                except:
                    # Fallback to window scroll
                    await self.page.evaluate('window.scrollBy(0, window.innerHeight)')
                
                await asyncio.sleep(self.wait_time)
                
                # Check intercepted responses for comment API calls
                responses = self.get_responses('api/comment/list')
                
                # Check if we got any new responses
                new_responses = [r for r, _ in responses if r.url not in processed_urls]
                if not new_responses:
                    scrolls_without_response += 1
                    if scrolls_without_response >= 10:
                        logger.warning(f"No API responses intercepted after 10 scrolls")
                        scrolls_without_response = 0
                else:
                    scrolls_without_response = 0
                
                comments_before = len(comments)
                
                for resp, json_data in responses:
                    # Skip already processed responses
                    if resp.url in processed_urls:
                        continue
                    processed_urls.add(resp.url)
                    
                    if json_data and 'comments' in json_data:
                        comment_list = json_data['comments']
                        logger.debug(f"Processing response with {len(comment_list)} comments")
                        
                        for item in comment_list:
                            if not isinstance(item, dict):
                                continue
                            
                            comment = self._parse_comment_data(item, video_id)
                            if comment and comment.comment_id not in seen_ids:
                                seen_ids.add(comment.comment_id)
                                comments.append(comment)
                                logger.debug(f"Added comment {comment.comment_id}")
                
                # Check if we got new comments
                new_comments = len(comments) - comments_before
                if new_comments == 0:
                    no_new_comments_count += 1
                    if no_new_comments_count >= 5:
                        logger.info(f"No new comments after 5 scrolls, stopping")
                        break
                else:
                    no_new_comments_count = 0
                    logger.debug(f"Scroll {scroll_count}: +{new_comments} comments (total: {len(comments)})")
                
                scroll_count += 1
                
                if (scroll_count % 10 == 0):
                    logger.info(f"Scrolled {scroll_count} times, found {len(comments)} comments")
            
            logger.info(f"Total comments fetched: {len(comments)}")
            return comments
            
        except Exception as e:
            logger.error(f"Error getting video comments: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return comments  # Return what we have so far
    
    def _parse_comment_data(self, item: Dict, video_id: str) -> Optional[CommentData]:
        """Parse comment data from API response"""
        try:
            if not isinstance(item, dict):
                logger.debug(f"Skipping non-dict comment: {type(item)}")
                return None
            
            comment_id = str(item.get('cid', ''))
            if not comment_id:
                logger.debug("Skipping comment with no ID")
                return None
            
            user = item.get('user', {})
            if not isinstance(user, dict):
                user = {}
            
            create_time = int(item.get('create_time', 0))
            create_time_iso = datetime.fromtimestamp(create_time).isoformat() if create_time else ""
            
            return CommentData(
                comment_id=comment_id,
                video_id=video_id,
                author_username=user.get('nickname', ''),
                author_unique_id=user.get('unique_id', ''),
                text=item.get('text', ''),
                create_time=create_time,
                create_time_iso=create_time_iso,
                like_count=int(item.get('digg_count', 0)),
                reply_count=int(item.get('reply_comment_total', 0)),
                is_author_verified=bool(user.get('verified', False)),
                scraped_at=datetime.now().isoformat()
            )
        except Exception as e:
            logger.debug(f"Error parsing comment: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None


# ============================================================================
# MAIN CRAWLER
# ============================================================================

async def crawl_comments_from_last_n_videos(
    username: str,
    n_videos: int = DEFAULT_N_VIDEOS,
    headless: bool = None,
    proxy: str = None
) -> Dict[str, Any]:
    """
    Crawl comments from the N most recent videos of a TikTok account.
    
    Args:
        username: TikTok username (without @)
        n_videos: Number of recent videos to crawl comments from
        headless: Run browser in headless mode
        proxy: Proxy to use (default: use PROXY constant)
    
    Returns:
        Dictionary with results
    """
    if headless is None:
        headless = HEADLESS
    
    if proxy is None:
        proxy = PROXY
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(OUTPUT_BASE_DIR) / username / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info(f"CRAWLING COMMENTS FROM LAST {n_videos} VIDEOS")
    logger.info(f"Account: @{username}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)
    
    results = {
        "username": username,
        "n_videos": n_videos,
        "scraped_at": datetime.now().isoformat(),
        "videos_processed": [],
        "total_comments": 0,
        "errors": []
    }
    
    try:
        async with CommentScraper(
            headless=headless,
            slow_mo=SLOW_MO,
            proxy=proxy
        ) as scraper:
            
            # Step 1: Get last N videos
            logger.info(f"Step 1: Fetching last {n_videos} videos from @{username}...")
            videos = await scraper.get_user_videos(
                username=username,
                max_videos=n_videos,
                lookback_days=365
            )
            
            if not videos:
                error_msg = f"No videos found for @{username}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                return results
            
            # Sort by create time and take the N most recent
            videos_sorted = sorted(videos, key=lambda v: v.create_timestamp, reverse=True)
            videos_to_crawl = videos_sorted[:n_videos]
            
            logger.info(f"Found {len(videos)} videos, selecting {len(videos_to_crawl)} most recent")
            
            # Step 2: Crawl comments for each video
            for idx, video in enumerate(videos_to_crawl, 1):
                logger.info("="*80)
                logger.info(f"Step 2.{idx}: Crawling comments for video {video.video_id}")
                logger.info(f"Video date: {video.create_time}")
                logger.info(f"Views: {video.stats.get('playCount', 0):,}")
                logger.info(f"Expected comments: {video.stats.get('commentCount', 0):,}")
                logger.info("="*80)
                
                try:
                    comments = await scraper.get_video_comments(
                        video_id=video.video_id,
                        max_comments=10000
                    )
                    
                    # Save comments for this video
                    video_dir = output_dir / f"video_{video.video_id}"
                    video_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save comments JSON
                    comments_file = video_dir / "comments.json"
                    with open(comments_file, 'w', encoding='utf-8') as f:
                        json.dump([asdict(c) for c in comments], f, ensure_ascii=False, indent=2)
                    
                    # Save video info
                    video_info_file = video_dir / "video_info.json"
                    with open(video_info_file, 'w', encoding='utf-8') as f:
                        json.dump(asdict(video), f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"✓ Saved {len(comments)} comments to {comments_file}")
                    
                    # Add to results
                    results["videos_processed"].append({
                        "video_id": video.video_id,
                        "create_time": video.create_time,
                        "views": video.stats.get('playCount', 0),
                        "expected_comments": video.stats.get('commentCount', 0),
                        "crawled_comments": len(comments),
                        "comments_file": str(comments_file),
                        "video_info_file": str(video_info_file)
                    })
                    
                    results["total_comments"] += len(comments)
                    
                except Exception as e:
                    error_msg = f"Error crawling comments for video {video.video_id}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    continue
                
                # Delay between videos
                if idx < len(videos_to_crawl):
                    logger.info("Waiting 3 seconds before next video...")
                    await asyncio.sleep(3)
            
            # Save overall summary
            summary_file = output_dir / "summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info("="*80)
            logger.info("CRAWLING COMPLETED")
            logger.info(f"Videos processed: {len(results['videos_processed'])}/{n_videos}")
            logger.info(f"Total comments: {results['total_comments']:,}")
            logger.info(f"Summary saved to: {summary_file}")
            logger.info("="*80)
    
    except Exception as e:
        error_msg = f"Fatal error: {e}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        results["errors"].append(error_msg)
    
    return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Crawl comments from last N videos of a TikTok account',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crawl_last_n_videos_comments.py bac_hello2
  python crawl_last_n_videos_comments.py bac_hello2 --n 5
  python crawl_last_n_videos_comments.py bac_hello2 --n 3 --headless
  python crawl_last_n_videos_comments.py jenny.huynh._ --n 10
        """
    )
    
    parser.add_argument(
        'username',
        help='TikTok username (without @)'
    )
    
    parser.add_argument(
        '--n',
        type=int,
        default=DEFAULT_N_VIDEOS,
        help=f'Number of recent videos to crawl (default: {DEFAULT_N_VIDEOS})'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run browser in headless mode'
    )
    
    parser.add_argument(
        '--proxy',
        type=str,
        default=None,
        help='Proxy server (default: use hardcoded proxy)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'crawl_comments_{args.username}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                encoding='utf-8'
            )
        ]
    )
    
    # Run crawler
    results = asyncio.run(crawl_comments_from_last_n_videos(
        username=args.username,
        n_videos=args.n,
        headless=args.headless,
        proxy=args.proxy
    ))
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Username: @{args.username}")
    print(f"Videos processed: {len(results['videos_processed'])}/{args.n}")
    print(f"Total comments: {results['total_comments']:,}")
    if results['errors']:
        print(f"\nErrors encountered: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")
    print("="*80)


if __name__ == "__main__":
    # Setup module logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('TikTok.CommentCrawler')
    
    main()
