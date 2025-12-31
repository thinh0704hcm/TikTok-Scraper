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
    
    async def start(self):
        """Start the browser with maximized window"""
        logger.info("Starting Playwright browser...")
        
        from playwright.async_api import async_playwright
        
        self.playwright = await async_playwright().start()
        
        launch_args = {
            'headless': self.headless,
            'slow_mo': self.slow_mo,
            'args': [
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-web-resources',
                '--start-maximized',  # Start maximized
            ]
        }
        
        # Add proxy if provided
        if self.proxy_server:
            logger.info(f"Using proxy: {self.proxy_server}")
            launch_args['proxy'] = {'server': self.proxy_server}
        elif self.proxy:
            logger.info(f"Using proxy: {self.proxy}")
            launch_args['proxy'] = {'server': self.proxy}
        
        self.browser = await self.playwright.chromium.launch(**launch_args)
        
        context_args = {
            'viewport': None,  # No viewport = use full window size
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'locale': 'en-US',
            'no_viewport': True,  # Allow browser to use actual window size
        }
        
        # Add proxy with credentials to context if parsed
        if self.proxy_server:
            context_args['proxy'] = {'server': self.proxy_server}
            if self.proxy_username and self.proxy_password:
                context_args['http_credentials'] = {
                    'username': self.proxy_username,
                    'password': self.proxy_password,
                }
                logger.info("Proxy credentials added to context")
        
        self.context = await self.browser.new_context(**context_args)
        
        # Anti-detection
        await self.context.add_init_script("""
            () => {
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            }
        """)
        
        self.page = await self.context.new_page()
        
        # Set up request/response interception
        self.page.on("request", self._on_request)
        self.page.on("response", self._on_response)
        
        logger.info("Navigating to TikTok...")
        await self.page.goto('https://www.tiktok.com/', wait_until='domcontentloaded', timeout=45000)
        await asyncio.sleep(3)
        
        await self._update_cookies()
        logger.info("Browser ready!")
    
    async def get_pinned_video_ids(self, username: str) -> List[str]:
        """
        Get list of pinned video IDs for a user by checking their profile.
        
        Args:
            username: TikTok username
            
        Returns:
            List of pinned video IDs
        """
        pinned_ids = []
        try:
            url = f"https://www.tiktok.com/@{username}"
            await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            await asyncio.sleep(3)
            
            # Find all video items with pinned badge
            pinned_ids = await self.page.evaluate('''
                () => {
                    const pinnedVideos = [];
                    // Find all video items
                    const videoItems = document.querySelectorAll('[data-e2e="user-post-item"]');
                    
                    for (const item of videoItems) {
                        // Check if this item has a pinned badge
                        const badge = item.querySelector('[data-e2e="video-card-badge"], [class*="DivBadge"]');
                        if (badge && badge.textContent.toLowerCase().includes('pinned')) {
                            // Get video link to extract ID
                            const link = item.querySelector('a[href*="/video/"]');
                            if (link) {
                                const href = link.getAttribute('href');
                                const match = href.match(/\\/video\\/(\\d+)/);
                                if (match) {
                                    pinnedVideos.push(match[1]);
                                }
                            }
                        }
                    }
                    return pinnedVideos;
                }
            ''')
            
            if pinned_ids:
                logger.info(f"Found {len(pinned_ids)} pinned videos: {pinned_ids}")
            
        except Exception as e:
            logger.debug(f"Error getting pinned videos: {e}")
        
        return pinned_ids
    
    async def get_video_comments(
        self,
        video_id: str,
        username: str,
        max_comments: int = 10000
    ) -> List[CommentData]:
        """
        Get all comments from a specific video.
        
        Args:
            video_id: TikTok video ID
            username: TikTok username (for constructing URL)
            max_comments: Maximum number of comments to fetch
        
        Returns:
            List of comment data
        """
        url = f"https://www.tiktok.com/@{username}/video/{video_id}"
        
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
            
            # Check for CAPTCHA
            page_content = await self.page.content()
            for captcha_text in self.CAPTCHA_LOCATORS:
                if captcha_text.lower() in page_content.lower():
                    logger.warning(f"⚠️  CAPTCHA detected: '{captcha_text}'")
                    logger.warning("Please solve the CAPTCHA manually...")
                    await asyncio.sleep(30)  # Give time to solve
                    break
            
            # Wait for video content and comment button to load
            try:
                await self.page.wait_for_selector('[data-e2e="browse-video"], video', timeout=15000)
                logger.info("✓ Video content loaded")
            except:
                logger.warning("Video content not found quickly, checking page...")
                current_url = self.page.url
                logger.info(f"Current URL: {current_url}")
            
            # Wait for comment button to be visible and clickable
            await asyncio.sleep(2)
            
            # Click on comment button to open comment section
            comment_button_clicked = False
            comment_button_selectors = [
                'button[aria-label*="comment" i]',
                'button[aria-label*="Comment" i]',
                '[data-e2e="comment-icon"]',
                '[data-e2e="comment-count"]',
                'button:has([data-e2e="comment-icon"])',
                'button:has([data-e2e="comment-count"])',
            ]
            
            for selector in comment_button_selectors:
                try:
                    comment_button = await self.page.wait_for_selector(selector, timeout=5000)
                    if comment_button:
                        logger.info(f"✓ Found comment button with selector: {selector}")
                        await comment_button.click()
                        comment_button_clicked = True
                        logger.info("✓ Clicked on comment button to open comment section")
                        await asyncio.sleep(3)  # Wait for comment panel to open
                        break
                except Exception as e:
                    logger.debug(f"Comment button not found with selector {selector}: {e}")
                    continue
            
            if not comment_button_clicked:
                logger.warning("Could not find/click comment button, trying alternative approach...")
                # Try clicking on any element that contains comment info
                try:
                    await self.page.evaluate('''
                        () => {
                            const buttons = document.querySelectorAll('button');
                            for (const btn of buttons) {
                                if (btn.getAttribute('aria-label') && 
                                    btn.getAttribute('aria-label').toLowerCase().includes('comment')) {
                                    btn.click();
                                    return true;
                                }
                            }
                            return false;
                        }
                    ''')
                    await asyncio.sleep(3)
                except Exception as e:
                    logger.debug(f"Alternative click failed: {e}")
            
            # Now wait for comment list to appear
            await asyncio.sleep(2)
            
            # Try to find comment container
            comment_container = None
            comment_selectors = [
                '[class*="DivCommentListContainer"]',
                '[class*="CommentListContainer"]',
            ]
            
            for selector in comment_selectors:
                try:
                    comment_container = await self.page.wait_for_selector(selector, timeout=5000)
                    if comment_container:
                        logger.info(f"✓ Comment section found with selector: {selector}")
                        break
                except:
                    continue
            
            if not comment_container:
                logger.warning("Comment section container not found after clicking button")
            
            # Parse comments from DOM instead of API responses
            logger.info("Scrolling comment section until all comments are loaded...")
            
            # Scroll to load ALL comments first using wheel events, then parse
            scroll_count = 0
            max_scrolls = 200
            no_scroll_progress_count = 0
            last_comment_count = 0
            
            while scroll_count < max_scrolls:
                # Get container bounding box to position mouse correctly
                container_info = await self.page.evaluate('''
                    () => {
                        const commentContainer = document.querySelector(
                            '[class*="DivCommentListContainer"]'
                        );
                        
                        if (!commentContainer) {
                            return { found: false, error: 'Container not found' };
                        }
                        
                        const rect = commentContainer.getBoundingClientRect();
                        const commentItems = document.querySelectorAll('[class*="DivVirtualItemContainer"]');
                        
                        return {
                            found: true,
                            x: rect.left + rect.width / 2,
                            y: rect.top + rect.height / 2,
                            scrollTop: commentContainer.scrollTop,
                            scrollHeight: commentContainer.scrollHeight,
                            clientHeight: commentContainer.clientHeight,
                            commentCount: commentItems.length
                        };
                    }
                ''')
                
                if not container_info.get('found'):
                    logger.error(f"Comment container not found: {container_info.get('error')}")
                    break
                
                current_comment_count = container_info.get('commentCount', 0)
                max_scroll = container_info['scrollHeight'] - container_info['clientHeight']
                is_at_bottom = container_info['scrollTop'] >= max_scroll - 10
                
                if scroll_count == 0:
                    logger.info(f"Starting scroll - Found {current_comment_count} comments initially")
                
                # Move mouse to comment container and scroll with wheel
                await self.page.mouse.move(container_info['x'], container_info['y'])
                await self.page.mouse.wheel(0, 500)  # Scroll down 500px
                
                # Wait for content to load
                await asyncio.sleep(0.8)
                
                # Check if we got more comments
                if current_comment_count > last_comment_count:
                    no_scroll_progress_count = 0
                    last_comment_count = current_comment_count
                else:
                    no_scroll_progress_count += 1
                
                # Log progress periodically
                if scroll_count % 10 == 0:
                    logger.info(f"Scroll {scroll_count}: {current_comment_count} comments loaded")
                
                # Check if we're done
                if is_at_bottom and no_scroll_progress_count >= 3:
                    logger.info(f"Reached bottom of comments after {scroll_count} scrolls")
                    break
                
                if no_scroll_progress_count >= 5:
                    logger.info(f"No new comments after 5 scrolls, assuming all comments loaded")
                    break
                
                scroll_count += 1
            
            logger.info(f"Finished scrolling after {scroll_count} scrolls. Now parsing all comments from DOM...")
            
            # Now parse ALL comments from DOM at once
            dom_comments = await self.page.evaluate('''
                () => {
                    const comments = [];
                    
                    // Find all comment items using the virtual item container class
                    const commentItems = document.querySelectorAll('[class*="DivVirtualItemContainer"]');
                    
                    console.log('Total comment items found:', commentItems.length);
                    
                    let index = 0;
                    for (const item of commentItems) {
                        try {
                            // Extract author username
                            let authorUsername = '';
                            let authorUniqueId = '';
                            
                            // Look for author link
                            const authorLink = item.querySelector('a[href*="/@"]');
                            if (authorLink) {
                                authorUsername = authorLink.textContent?.trim() || '';
                                const href = authorLink.getAttribute('href') || '';
                                const match = href.match(/@([^/?]+)/);
                                if (match) authorUniqueId = match[1];
                            }
                            
                            // Extract comment text - look for span with text content
                            let commentText = '';
                            
                            // Try to find text spans (usually comment text is in a span)
                            const textSpans = item.querySelectorAll('span[data-e2e="comment-level-1"] span, span');
                            for (const span of textSpans) {
                                const text = span.textContent?.trim() || '';
                                // Filter out short texts (likely buttons/labels)
                                if (text.length > 1 && text.length > commentText.length && 
                                    text !== authorUsername &&
                                    !text.match(/^\\d+[KMB]?$/) && // not just numbers (likes)
                                    !text.match(/^\\d+[dhmw]/) && // not time ago
                                    text !== 'Reply' &&
                                    text !== 'View' &&
                                    text !== 'View more') {
                                    commentText = text;
                                }
                            }
                            
                            // Extract time
                            let timeText = '';
                            const allText = item.textContent || '';
                            const timeMatch = allText.match(/(\\d+[dhmw] ago|[A-Z][a-z]+ \\d+)/);
                            if (timeMatch) timeText = timeMatch[1];
                            
                            // Generate unique ID
                            const commentId = 'dom_' + index;
                            
                            if (authorUsername && commentText && commentText.length > 0) {
                                comments.push({
                                    comment_id: commentId,
                                    author_username: authorUsername,
                                    author_unique_id: authorUniqueId || authorUsername.replace('@', ''),
                                    text: commentText.substring(0, 2000),
                                    time_text: timeText,
                                    like_count: 0,
                                    reply_count: 0,
                                    is_verified: false
                                });
                            }
                            
                            index++;
                        } catch (e) {
                            console.error('Error parsing comment:', e);
                        }
                    }
                    
                    console.log('Successfully parsed comments:', comments.length);
                    return comments;
                }
            ''')
            
            logger.info(f"Found {len(dom_comments)} comments in DOM")
            
            # Process DOM comments
            for dom_comment in dom_comments:
                comment_id = dom_comment['comment_id']
                if comment_id not in seen_ids:
                    seen_ids.add(comment_id)
                    
                    # Create CommentData object
                    comment = CommentData(
                        comment_id=comment_id,
                        video_id=video_id,
                        author_username=dom_comment['author_username'],
                        author_unique_id=dom_comment['author_unique_id'],
                        text=dom_comment['text'],
                        create_time=0,
                        create_time_iso=dom_comment['time_text'],
                        like_count=dom_comment['like_count'],
                        reply_count=dom_comment['reply_count'],
                        is_author_verified=dom_comment['is_verified'],
                        scraped_at=datetime.now().isoformat()
                    )
                    comments.append(comment)
            
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
        "pinned_videos_skipped": [],
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
            
            # Get pinned video IDs to skip them
            logger.info("Checking for pinned videos to skip...")
            pinned_video_ids = await scraper.get_pinned_video_ids(username)
            if pinned_video_ids:
                logger.info(f"Found {len(pinned_video_ids)} pinned videos to skip: {pinned_video_ids}")
            else:
                logger.info("No pinned videos found")
            
            # Sort by create time and filter out pinned videos
            videos_sorted = sorted(videos, key=lambda v: v.create_timestamp, reverse=True)
            
            # Filter out pinned videos
            non_pinned_videos = [v for v in videos_sorted if v.video_id not in pinned_video_ids]
            skipped_pinned = len(videos_sorted) - len(non_pinned_videos)
            if skipped_pinned > 0:
                logger.info(f"Skipped {skipped_pinned} pinned videos")
                results["pinned_videos_skipped"] = pinned_video_ids
            
            # Take the N most recent non-pinned videos
            videos_to_crawl = non_pinned_videos[:n_videos]
            
            logger.info(f"Found {len(videos)} videos, skipped {skipped_pinned} pinned, selecting {len(videos_to_crawl)} most recent non-pinned")
            
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
                        username=username,
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
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'logs/crawl_comments_{args.username}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
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
