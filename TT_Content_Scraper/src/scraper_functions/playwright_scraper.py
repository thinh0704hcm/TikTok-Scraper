"""
TikTok Profile Scraper with Browser Fingerprint Replay Support

NEW: Supports loading real device fingerprints for better anti-detection
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from patchright.async_api import async_playwright, Page, BrowserContext, Browser, Request, Response

logger = logging.getLogger('TikTok.ProfileScraper')


# ============================================================================
# FINGERPRINT LOADER
# ============================================================================

class BrowserFingerprint:
    """Load and manage browser fingerprints"""
    
    @staticmethod
    def load_from_file(filepath: str) -> Optional[Dict]:
        """Load fingerprint from JSON file"""
        try:
            with open(filepath, 'r') as f:
                fp = json.load(f)
                logger.info(f"✓ Loaded fingerprint from {filepath}")
                logger.info(f"  Device: {fp.get('platform', 'Unknown')}")
                logger.info(f"  GPU: {fp.get('webgl', {}).get('renderer', 'Unknown')[:60]}...")
                return fp
        except FileNotFoundError:
            logger.warning(f"Fingerprint file not found: {filepath}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid fingerprint JSON: {e}")
            return None
    
    @staticmethod
    def get_default_fingerprint() -> Dict:
        """Get default Windows fingerprint (fallback)"""
        return {
            "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "platform": "Win32",
            "hardwareConcurrency": 8,
            "deviceMemory": 8,
            "maxTouchPoints": 0,
            "languages": ["en-US", "en"],
            "webgl": {
                "vendor": "Google Inc. (Intel)",
                "renderer": "ANGLE (Intel, Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0, D3D11)"
            },
            "screen": {
                "width": 1920,
                "height": 1080,
                "availWidth": 1920,
                "availHeight": 1040,
                "colorDepth": 24,
                "pixelDepth": 24
            },
            "timezone": "America/New_York",
            "timezoneOffset": 300
        }


# ============================================================================
# DATA CLASSES (unchanged)
# ============================================================================

@dataclass
class VideoData:
    video_id: str
    author_username: str
    author_id: str
    description: str
    create_time: str
    create_timestamp: int
    hashtags: List[str]
    stats: Dict[str, Any]
    music_title: Optional[str] = None
    scraped_at: Optional[str] = None


@dataclass
class TimeSeriesPoint:
    period_start: str
    period_end: str
    timestamp: int
    video_count: int
    total_views: int
    total_likes: int
    total_comments: int
    total_shares: int
    total_collects: int
    avg_views_per_video: float
    avg_likes_per_video: float
    avg_comments_per_video: float
    avg_shares_per_video: float
    engagement_rate: float


@dataclass
class InterceptedRequest:
    url: str
    method: str
    headers: Dict[str, str]
    post_data: Optional[str] = None


@dataclass
class InterceptedResponse:
    url: str
    status: int
    headers: Dict[str, str]
    body: Optional[bytes] = None


# ============================================================================
# PLAYWRIGHT PROFILE SCRAPER WITH FINGERPRINT SUPPORT
# ============================================================================

class PlaywrightProfileScraper:
    """TikTok profile scraper with fingerprint replay"""
    
    CAPTCHA_LOCATORS = [
        'Rotate the shapes',
        'Verify to continue:',
        'Click on the shapes',
        'Drag the slider',
        'Select 2 objects that are the same',
    ]
    
    def __init__(
        self,
        output_dir: str = "data/profile_time_series/",
        headless: bool = False,
        slow_mo: int = 50,
        wait_time: float = 1.0,
        max_captcha_wait: int = 120,
        proxy: Optional[str] = None,
        fingerprint_file: Optional[str] = None  # NEW: Fingerprint file path
    ):
        self.output_dir = Path(output_dir)
        self.headless = headless
        self.slow_mo = slow_mo
        self.wait_time = wait_time
        self.max_captcha_wait = max_captcha_wait
        self.proxy = proxy
        
        # NEW: Load fingerprint
        if fingerprint_file and Path(fingerprint_file).exists():
            self.fingerprint = BrowserFingerprint.load_from_file(fingerprint_file)
        else:
            if fingerprint_file:
                logger.warning(f"Fingerprint file not found: {fingerprint_file}, using default")
            self.fingerprint = BrowserFingerprint.get_default_fingerprint()
        
        # Parse proxy credentials
        self.proxy_server = None
        self.proxy_username = None
        self.proxy_password = None
        if proxy:
            self._parse_proxy_url(proxy)
        
        self.playwright = None
        self.browser: Browser = None
        self.context: BrowserContext = None
        self.page: Page = None
        
        self._requests: List[InterceptedRequest] = []
        self._responses: List[InterceptedResponse] = []
        
        self.ms_token: str = None
        self.cookies: Dict[str, str] = {}
        
        self.stats = {
            "videos_scraped": 0,
            "profiles_scraped": 0,
            "errors": 0,
            "api_calls": 0,
        }
    
    def _parse_proxy_url(self, proxy_url: str):
        """Parse proxy URL and extract credentials"""
        try:
            if proxy_url.count(":") == 3 and '@' not in proxy_url:
                ip, port, username, password = proxy_url.strip().split(":")
                self.proxy_username = username
                self.proxy_password = password
                self.proxy_server = f"http://{ip}:{port}"
                logger.info(f"Proxy: {self.proxy_server}")
            else:
                from urllib.parse import urlparse
                parsed = urlparse(proxy_url if '://' in proxy_url else f'http://{proxy_url}')
                if parsed.username and parsed.password:
                    self.proxy_username = parsed.username
                    self.proxy_password = parsed.password
                if parsed.port:
                    self.proxy_server = f"{parsed.scheme or 'http'}://{parsed.hostname}:{parsed.port}"
                else:
                    self.proxy_server = f"{parsed.scheme or 'http'}://{parsed.hostname}:8080"
                logger.info(f"Proxy: {self.proxy_server}")
        except Exception as e:
            logger.warning(f"Failed to parse proxy: {e}")
            self.proxy_server = proxy_url
    
    # ========================================================================
    # BROWSER LIFECYCLE WITH FINGERPRINT INJECTION
    # ========================================================================
    
    async def _on_request(self, request: Request):
        try:
            self._requests.append(InterceptedRequest(
                url=request.url,
                method=request.method,
                headers=dict(request.headers),
                post_data=request.post_data
            ))
        except Exception as e:
            logger.debug(f"Error storing request: {e}")
    
    async def _on_response(self, response: Response):
        try:
            if 'api/post/item_list' in response.url or 'api/user/detail' in response.url:
                body = None
                try:
                    body = await response.body()
                except:
                    pass
                self._responses.append(InterceptedResponse(
                    url=response.url,
                    status=response.status,
                    headers=dict(response.headers),
                    body=body
                ))
                logger.debug(f"Intercepted API: {response.url[:100]}")
        except Exception as e:
            logger.debug(f"Error storing response: {e}")
    
    def get_responses(self, pattern: str) -> List[Tuple[InterceptedResponse, Optional[Dict]]]:
        results = []
        for r in self._responses:
            if pattern in r.url:
                json_data = None
                if r.body:
                    try:
                        json_data = json.loads(r.body.decode('utf-8'))
                    except:
                        pass
                results.append((r, json_data))
        return results
    
    def clear_responses(self):
        self._responses = []
    
    def clear_requests(self):
        self._requests = []
    
    async def start(self):
        """Start browser with fingerprint injection"""
        import os
        import sys
        import json # Ensure json is imported
        import random
        
        logger.info("Starting Patchright browser with fingerprint replay...")
        logger.info(f"Mode: {'HEADLESS' if self.headless else 'VISIBLE'}")
        logger.info(f"Platform: {sys.platform}")
        
        if not self.headless and sys.platform.startswith('linux'):
            display = os.environ.get('DISPLAY', '')
            if display:
                logger.info(f"Display: {display}")
        
        self.playwright = await async_playwright().start()
        
        # Use fingerprint's user agent
        user_agent = self.fingerprint.get('userAgent', 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')
        
        launch_args = {
            'headless': self.headless,
            'slow_mo': self.slow_mo,
            'args': [
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--use-angle=swiftshader',
                '--use-gl=angle',
                '--disable-gpu',
                '--disable-infobars', # Added from your test script
                '--exclude-switches=enable-automation', # Added from your test script
                '--disable-automation', # Added from your test script
            ],
            'ignore_default_args': ['--enable-automation'], # Added from your test script
            'chromium_sandbox': False # Added from your test script
        }
        
        if self.proxy_server:
            logger.info(f"Proxy: {self.proxy_server}")
            launch_args['proxy'] = {'server': self.proxy_server}
        elif self.proxy:
            launch_args['proxy'] = {'server': self.proxy}
        
        self.browser = await self.playwright.chromium.launch(**launch_args)
        
        # Use fingerprint for context
        screen = self.fingerprint.get('screen', {'width': 1920, 'height': 1080})
        
        context_args = {
            'viewport': {'width': screen['width'], 'height': screen['height']},
            'screen': {'width': screen['width'], 'height': screen['height']},
            'user_agent': user_agent,
            'locale': 'en-US',
            'timezone_id': self.fingerprint.get('timezone', 'America/New_York'),
            'color_scheme': 'light',
            'has_touch': False,
            'is_mobile': False,
            'device_scale_factor': 1,
            'extra_http_headers': {
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
            }
        }
        
        if self.proxy_server:
            context_args['proxy'] = {'server': self.proxy_server}
            if self.proxy_username and self.proxy_password:
                context_args['http_credentials'] = {
                    'username': self.proxy_username,
                    'password': self.proxy_password,
                }
        
        self.context = await self.browser.new_context(**context_args)
        
        # --- ORIGINAL FINGERPRINT INJECTION (WebGL, Hardware, etc.) ---
        # Note: I removed the weaker webdriver deletion from here since we use CDP below
        fingerprint_json = json.dumps(self.fingerprint)
        await self.context.add_init_script(f"""
            (() => {{
                'use strict';
                
                const FINGERPRINT = {fingerprint_json};
                
                // WebGL spoofing with fingerprint data
                const webgl = FINGERPRINT.webgl || {{}};
                const VENDOR = webgl.vendor || 'Google Inc. (Intel)';
                const RENDERER = webgl.renderer || 'ANGLE (Intel, Intel(R) UHD Graphics 630)';
                
                const origGetParam = WebGLRenderingContext.prototype.getParameter;
                WebGLRenderingContext.prototype.getParameter = function(p) {{
                    if (p === 37445 || p === 7936) return VENDOR;
                    if (p === 37446 || p === 7937) return RENDERER;
                    return origGetParam.call(this, p);
                }};
                
                if (typeof WebGL2RenderingContext !== 'undefined') {{
                    const origGetParam2 = WebGL2RenderingContext.prototype.getParameter;
                    WebGL2RenderingContext.prototype.getParameter = function(p) {{
                        if (p === 37445 || p === 7936) return VENDOR;
                        if (p === 37446 || p === 7937) return RENDERER;
                        return origGetParam2.call(this, p);
                    }};
                }}
                
                const origGetExt = WebGLRenderingContext.prototype.getExtension;
                WebGLRenderingContext.prototype.getExtension = function(name) {{
                    const ext = origGetExt.call(this, name);
                    if (name === 'WEBGL_debug_renderer_info' && ext) {{
                        Object.defineProperty(ext, 'UNMASKED_VENDOR_WEBGL', {{value: 37445}});
                        Object.defineProperty(ext, 'UNMASKED_RENDERER_WEBGL', {{value: 37446}});
                    }}
                    return ext;
                }};
                
                // Platform
                Object.defineProperty(navigator, 'platform', {{
                    get: () => FINGERPRINT.platform || 'Win32',
                    configurable: true
                }});
                
                // Hardware
                Object.defineProperty(navigator, 'hardwareConcurrency', {{
                    get: () => FINGERPRINT.hardwareConcurrency || 8,
                    configurable: true
                }});
                
                if (FINGERPRINT.deviceMemory) {{
                    Object.defineProperty(navigator, 'deviceMemory', {{
                        get: () => FINGERPRINT.deviceMemory,
                        configurable: true
                    }});
                }}
                
                if (FINGERPRINT.maxTouchPoints !== undefined) {{
                    Object.defineProperty(navigator, 'maxTouchPoints', {{
                        get: () => FINGERPRINT.maxTouchPoints,
                        configurable: true
                    }});
                }}
                
                // Languages
                Object.defineProperty(navigator, 'languages', {{
                    get: () => FINGERPRINT.languages || ['en-US', 'en'],
                    configurable: true
                }});
                
                // Screen properties
                if (FINGERPRINT.screen) {{
                    const s = FINGERPRINT.screen;
                    Object.defineProperties(screen, {{
                        width: {{get: () => s.width, configurable: true}},
                        height: {{get: () => s.height, configurable: true}},
                        availWidth: {{get: () => s.availWidth || s.width, configurable: true}},
                        availHeight: {{get: () => s.availHeight || s.height - 40, configurable: true}},
                        colorDepth: {{get: () => s.colorDepth || 24, configurable: true}},
                        pixelDepth: {{get: () => s.pixelDepth || 24, configurable: true}}
                    }});
                }}

                // Chrome object
                Object.defineProperty(window, 'chrome', {{
                    get: () => ({{runtime: {{}}, app: {{}}, loadTimes: () => {{}}, csi: () => {{}}}}),
                    configurable: true
                }});
                
                console.log('%c✓ Fingerprint Loaded', 'color: green; font-weight: bold');
            }})();
        """)
        
        self.page = await self.context.new_page()

        # --- NEW: CDP METHOD FOR DEEP WEBDRIVER MASKING ---
        # This is inserted immediately after page creation as requested
        try:
            cdp = await self.context.new_cdp_session(self.page)
            await cdp.send('Page.addScriptToEvaluateOnNewDocument', {
                'source': """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined,
                        enumerable: false,
                        configurable: true
                    });
                    
                    delete navigator.__proto__.webdriver;
                    
                    // Hide from all inspection methods
                    const originalKeys = Object.keys;
                    Object.keys = function(obj) {
                        if (obj === navigator || obj === Navigator.prototype) {
                            return originalKeys(obj).filter(k => k !== 'webdriver');
                        }
                        return originalKeys(obj);
                    };
                    
                    const originalGetOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
                    Object.getOwnPropertyDescriptor = function(obj, prop) {
                        if ((obj === navigator || obj === Navigator.prototype) && prop === 'webdriver') {
                            return undefined;
                        }
                        return originalGetOwnPropertyDescriptor(obj, prop);
                    };
                """
            })
            logger.info("✓ CDP Session established and webdriver masking script injected")
        except Exception as e:
            logger.error(f"Failed to initialize CDP session: {e}")
        
        # Set up interception
        self.page.on("request", self._on_request)
        self.page.on("response", self._on_response)
        
        logger.info("Navigating to TikTok...")
        await asyncio.sleep(random.uniform(1.0, 2.5))
        
        await self.page.goto('https://www.tiktok.com/', 
                             wait_until='domcontentloaded', 
                             timeout=45000)
        
        await asyncio.sleep(random.uniform(2.5, 4.0))
        await self._update_cookies()
        logger.info("✓ Browser ready with fingerprint!")
    
    async def stop(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def _update_cookies(self):
        browser_cookies = await self.context.cookies()
        for cookie in browser_cookies:
            self.cookies[cookie['name']] = cookie['value']
            if cookie['name'] == 'msToken':
                self.ms_token = cookie['value']
    
    async def _extract_page_data(self) -> Optional[Dict]:
        try:
            data = await self.page.evaluate('''
                () => {
                    if (window.__UNIVERSAL_DATA_FOR_REHYDRATION__) 
                        return window.__UNIVERSAL_DATA_FOR_REHYDRATION__;
                    if (window.SIGI_STATE) 
                        return window.SIGI_STATE;
                    return null;
                }
            ''')
            return data
        except Exception as e:
            logger.debug(f"Error extracting page data: {e}")
            return None
    # ========================================================================
    # VIDEO SCRAPING (Playwright scrolling method)
    # ========================================================================
    
    async def get_user_videos(
        self,
        username: str,
        max_videos: int = 1000,
        lookback_days: int = 365
    ) -> List[VideoData]:
        """
        Get all videos from a user within the lookback period using Playwright scrolling.
        
        Args:
            username: TikTok username (without @)
            max_videos: Maximum number of videos to fetch
            lookback_days: How many days back to look
        """
        cutoff_timestamp = int((datetime.now() - timedelta(days=lookback_days)).timestamp())
        url = f"https://www.tiktok.com/@{username}"
        
        videos = []
        seen_ids = set()
        
        try:
            # Clear previous responses
            self.clear_responses()
            
            logger.info(f"Navigating to @{username}...")
            
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
            
            await asyncio.sleep(self.wait_time + 1)
            
            # Wait for video grid
            try:
                await self.page.wait_for_selector('[data-e2e="user-post-item"]', timeout=10000)
            except:
                logger.warning("Video grid not found, continuing anyway...")
            
            # Extract initial videos from page data
            data = await self._extract_page_data()
            if data:
                initial_videos = self._extract_videos_from_page_data(data, username, cutoff_timestamp)
                for video in initial_videos:
                    if video.video_id not in seen_ids:
                        seen_ids.add(video.video_id)
                        videos.append(video)
                        self.stats["videos_scraped"] += 1
                
                logger.info(f"Extracted {len(initial_videos)} videos from page data")
            
            # Scroll to trigger API calls and load more videos
            scroll_count = 0
            max_scrolls = 100
            no_new_videos_count = 0
            processed_urls = set()  # Track which responses we've processed
            scrolls_without_response = 0  # Track if we're getting any API responses
            
            while len(videos) < max_videos and scroll_count < max_scrolls:
                # Scroll down
                await self.page.evaluate('window.scrollBy(0, window.innerHeight)')
                await asyncio.sleep(self.wait_time)
                
                # Check intercepted responses
                responses = self.get_responses('api/post/item_list')
                
                # Check if we got any new responses
                new_responses = [r for r, _ in responses if r.url not in processed_urls]
                if not new_responses:
                    scrolls_without_response += 1
                    if scrolls_without_response >= 10:
                        logger.warning(f"No API responses intercepted after 10 scrolls - may need to scroll manually or check network")
                        scrolls_without_response = 0  # Reset to avoid spam
                else:
                    scrolls_without_response = 0
                
                videos_before = len(videos)
                old_videos_count = 0
                new_videos_this_scroll = 0
                
                for resp, json_data in responses:
                    # Skip already processed responses
                    if resp.url in processed_urls:
                        continue
                    processed_urls.add(resp.url)
                    
                    if json_data and 'itemList' in json_data:
                        items = json_data['itemList']
                        logger.debug(f"Processing response with {len(items)} items")
                        
                        for item in items:
                            if not isinstance(item, dict):
                                logger.debug(f"Skipping non-dict item type: {type(item)}")
                                continue
                                
                            video = self._parse_video_data(item, username)
                            if video and video.video_id not in seen_ids:
                                seen_ids.add(video.video_id)
                                
                                # Check if within lookback period
                                if video.create_timestamp >= cutoff_timestamp:
                                    videos.append(video)
                                    self.stats["videos_scraped"] += 1
                                    new_videos_this_scroll += 1
                                    logger.debug(f"Added video {video.video_id} from {video.create_time}")
                                else:
                                    old_videos_count += 1
                                    logger.debug(f"Skipped old video {video.video_id} from {video.create_time}")
                    else:
                        if json_data:
                            logger.debug(f"Response has no itemList. Keys: {list(json_data.keys()) if isinstance(json_data, dict) else 'not a dict'}")
                
                # Check if we got new videos
                new_videos = len(videos) - videos_before
                if new_videos == 0:
                    no_new_videos_count += 1
                    if no_new_videos_count >= 5:
                        logger.info(f"No new videos in date range after 5 scrolls, stopping")
                        logger.info(f"Total skipped (too old): {old_videos_count} videos")
                        break
                else:
                    no_new_videos_count = 0
                    logger.debug(f"Scroll {scroll_count}: +{new_videos} videos (total: {len(videos)})")
                
                scroll_count += 1
                
                if (scroll_count % 10 == 0):
                    logger.info(f"Scrolled {scroll_count} times, found {len(videos)} videos in date range")
            
            logger.info(f"Total videos fetched: {len(videos)} (within {lookback_days} days)")
            return videos
            
        except Exception as e:
            logger.error(f"Error getting user videos: {e}")
            self.stats["errors"] += 1
            return videos  # Return what we have so far
    
    def _extract_videos_from_page_data(
        self, 
        data: Dict, 
        username: str, 
        cutoff_timestamp: int
    ) -> List[VideoData]:
        """Extract videos from page data"""
        videos = []
        
        try:
            default_scope = data.get('__DEFAULT_SCOPE__', {})
            user_post = default_scope.get('webapp.user-post', {})
            item_list = user_post.get('itemList', [])
            
            # itemList sometimes contains just video IDs (strings), not full objects
            # Filter to only process dictionaries
            if item_list and isinstance(item_list, list):
                item_list = [item for item in item_list if isinstance(item, dict)]
            
            if not item_list:
                # Try ItemModule (older format)
                item_module = data.get('ItemModule', {})
                if item_module and isinstance(item_module, dict):
                    item_list = [v for v in item_module.values() if isinstance(v, dict)]
            
            logger.debug(f"Found {len(item_list)} video objects in page data")
            
            for item in item_list:
                if not isinstance(item, dict):
                    continue
                    
                video = self._parse_video_data(item, username)
                if video and video.create_timestamp >= cutoff_timestamp:
                    videos.append(video)
                    logger.debug(f"Extracted video {video.video_id} from page data")
        
        except Exception as e:
            logger.debug(f"Error extracting videos from page data: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return videos
    
    def _parse_video_data(self, item: Dict, username: str) -> Optional[VideoData]:
        """Parse video data from API response"""
        try:
            # Ensure item is a dictionary
            if not isinstance(item, dict):
                logger.debug(f"Skipping non-dict item: {type(item)}")
                return None
            
            author = item.get('author', {})
            if not isinstance(author, dict):
                author = {}
            
            stats = item.get('statsV2') or item.get('stats', {})
            if not isinstance(stats, dict):
                stats = {}
            
            normalized_stats = {
                'playCount': int(stats.get('playCount', 0) or 0),
                'diggCount': int(stats.get('diggCount', 0) or 0),
                'commentCount': int(stats.get('commentCount', 0) or 0),
                'shareCount': int(stats.get('shareCount', 0) or 0),
                'collectCount': int(stats.get('collectCount', 0) or 0),
            }
            
            hashtags = []
            challenges = item.get('challenges', []) or item.get('textExtra', [])
            if isinstance(challenges, list):
                for tag in challenges:
                    if isinstance(tag, dict):
                        name = tag.get('title') or tag.get('hashtagName', '')
                        if name:
                            hashtags.append(name)
            
            create_time = int(item.get('createTime', 0))
            create_time_iso = datetime.fromtimestamp(create_time).isoformat() if create_time else ""
            
            music = item.get('music', {})
            if not isinstance(music, dict):
                music = {}
            
            video_id = str(item.get('id', ''))
            if not video_id:
                logger.debug("Skipping video with no ID")
                return None
            
            return VideoData(
                video_id=video_id,
                author_username=username,
                author_id=str(author.get('id', '')),
                description=item.get('desc', ''),
                create_time=create_time_iso,
                create_timestamp=create_time,
                hashtags=hashtags,
                stats=normalized_stats,
                music_title=music.get('title', '') if music else None,
                scraped_at=datetime.now().isoformat()
            )
        except Exception as e:
            logger.debug(f"Error parsing video: {e}")
            return None
    
    # ========================================================================
    # TIME SERIES GENERATION
    # ========================================================================
    
    def create_time_series(
        self,
        videos: List[VideoData]
    ) -> List[TimeSeriesPoint]:
        """
        Aggregate video stats into time series (daily aggregation).
        
        Args:
            videos: List of video data
        
        Returns:
            List of time series points
        """
        if not videos:
            return []
        
        # Sort videos by timestamp
        videos_sorted = sorted(videos, key=lambda v: v.create_timestamp)
        
        # Group videos by day
        period_groups = defaultdict(list)
        
        for video in videos_sorted:
            dt = datetime.fromtimestamp(video.create_timestamp)
            period_key = dt.strftime('%Y-%m-%d')
            period_groups[period_key].append(video)
        
        # Create time series points
        time_series = []
        
        for period_key in sorted(period_groups.keys()):
            videos_in_period = period_groups[period_key]
            
            total_views = sum(v.stats['playCount'] for v in videos_in_period)
            total_likes = sum(v.stats['diggCount'] for v in videos_in_period)
            total_comments = sum(v.stats['commentCount'] for v in videos_in_period)
            total_shares = sum(v.stats['shareCount'] for v in videos_in_period)
            total_collects = sum(v.stats['collectCount'] for v in videos_in_period)
            
            video_count = len(videos_in_period)
            
            # Calculate averages
            avg_views = total_views / video_count if video_count > 0 else 0
            avg_likes = total_likes / video_count if video_count > 0 else 0
            avg_comments = total_comments / video_count if video_count > 0 else 0
            avg_shares = total_shares / video_count if video_count > 0 else 0
            
            # Engagement rate: (likes + comments + shares) / views
            total_engagement = total_likes + total_comments + total_shares
            engagement_rate = (total_engagement / total_views * 100) if total_views > 0 else 0
            
            # Get period boundaries
            first_video = videos_in_period[0]
            last_video = videos_in_period[-1]
            
            time_series.append(TimeSeriesPoint(
                period_start=first_video.create_time,
                period_end=last_video.create_time,
                timestamp=first_video.create_timestamp,
                video_count=video_count,
                total_views=total_views,
                total_likes=total_likes,
                total_comments=total_comments,
                total_shares=total_shares,
                total_collects=total_collects,
                avg_views_per_video=round(avg_views, 2),
                avg_likes_per_video=round(avg_likes, 2),
                avg_comments_per_video=round(avg_comments, 2),
                avg_shares_per_video=round(avg_shares, 2),
                engagement_rate=round(engagement_rate, 2)
            ))
        
        return time_series
    
    # ========================================================================
    # HIGH-LEVEL SCRAPING
    # ========================================================================
    
    async def scrape_user_time_series(
        self,
        username: str,
        max_videos: int = 1000,
        lookback_days: int = 365
    ) -> Dict[str, Any]:
        """
        Scrape user profile and create time series dataset (daily aggregation).
        
        Args:
            username: TikTok username (without @)
            max_videos: Maximum videos to fetch
            lookback_days: Days to look back (default 365 = 1 year)
        
        Returns:
            Dictionary with videos, time series, and metadata
        """
        logger.info(f"Starting time series scrape for @{username}")
        logger.info(f"Lookback: {lookback_days} days")
        
        # Calculate date range
        now = datetime.now()
        cutoff_date = now - timedelta(days=lookback_days)
        logger.info(f"Date range: {cutoff_date.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}")
        
        # Create output directory
        username_dir = self.output_dir / username
        username_dir.mkdir(parents=True, exist_ok=True)
        
        # Fetch all videos
        videos = await self.get_user_videos(username, max_videos, lookback_days)
        
        if not videos:
            logger.error("No videos fetched")
            return {"error": "No videos found"}
        
        logger.info(f"Fetched {len(videos)} videos")
        
        # Save raw videos
        videos_file = username_dir / f"videos_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(videos_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(v) for v in videos], f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(videos)} videos to {videos_file}")
        
        # Generate summary
        summary = {
            "username": username,
            "lookback_days": lookback_days,
            "date_range": {
                "start": cutoff_date.strftime('%Y-%m-%d'),
                "end": now.strftime('%Y-%m-%d')
            },
            "scraped_at": datetime.now().isoformat(),
            "total_videos": len(videos),
            "date_range_actual": {
                "earliest": min(v.create_time for v in videos) if videos else None,
                "latest": max(v.create_time for v in videos) if videos else None,
            },
            "total_stats": {
                "total_views": sum(v.stats['playCount'] for v in videos),
                "total_likes": sum(v.stats['diggCount'] for v in videos),
                "total_comments": sum(v.stats['commentCount'] for v in videos),
                "total_shares": sum(v.stats['shareCount'] for v in videos),
            },
            "average_stats": {
                "avg_views": sum(v.stats['playCount'] for v in videos) / len(videos),
                "avg_likes": sum(v.stats['diggCount'] for v in videos) / len(videos),
                "avg_comments": sum(v.stats['commentCount'] for v in videos) / len(videos),
            },
            "files": {
                "videos": str(videos_file.name),
            },
            "scraper_stats": self.stats
        }
        
        # Save summary
        summary_file = username_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved summary to {summary_file}")
        
        return summary
    
    def _save_time_series_csv(self, time_series: List[TimeSeriesPoint], filepath: Path):
        """Save time series as CSV"""
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if not time_series:
                return
            
            fieldnames = list(asdict(time_series[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for ts in time_series:
                writer.writerow(asdict(ts))
        
        logger.info(f"Saved CSV to {filepath}")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    username = "example_user"  # Replace with actual username
    
    async with PlaywrightProfileScraper(
        output_dir="data/profile_time_series/",
        headless=False,
    ) as scraper:
        
        # Scrape user's 1-year history with daily aggregation
        summary = await scraper.scrape_user_time_series(
            username=username,
            period='daily',  # or 'weekly', 'monthly'
            max_videos=1000,
            lookback_days=365
        )
        
        print("\n" + "="*70)
        print("TIME SERIES SUMMARY")
        print("="*70)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())