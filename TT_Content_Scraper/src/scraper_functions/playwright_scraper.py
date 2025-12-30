"""
TikTok Profile Scraper with Network Interception (Brave Edition)

Features:
- Network interception for api/post/item_list to capture scrolled videos
- Milestone-based scraping (fetch videos after a specific datetime)
- Brave browser with fingerprint spoofing
- Pinned video filtering
- Console logging for debugging
- Cookie persistence
- Proxy support with authentication
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Page, BrowserContext, Browser, Response

logger = logging.getLogger('TikTok.Scraper')


# ============================================================================
# BRAVE FINDER
# ============================================================================

def find_brave():
    """Find Brave Browser on the system."""
    paths = [
        '/usr/bin/brave-browser',           # Linux standard
        '/usr/bin/brave',                   # Linux alternative
        '/opt/brave.com/brave/brave',       # Linux manual install
        'C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe', # Windows
        '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser' # MacOS
    ]
    for path in paths:
        if os.path.exists(path):
            logger.info(f"Found Brave: {path}")
            return path
    logger.warning("Brave not found, will use default Chromium")
    return None


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
# DATA CLASSES
# ============================================================================

@dataclass
class VideoData:
    """Data class for video information"""
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


# ============================================================================
# PLAYWRIGHT PROFILE SCRAPER WITH NETWORK INTERCEPTION
# ============================================================================

class PlaywrightScraper:
    """
    TikTok profile scraper with network interception.
    Captures both initial hydration data and scrolled API responses.
    """
    
    CAPTCHA_LOCATORS = [
        'Rotate the shapes',
        'Verify to continue:',
        'Click on the shapes',
        'Drag the slider',
        'Select 2 objects that are the same',
    ]
    
    def __init__(
        self,
        headless: bool = False,
        slow_mo: int = 50,
        wait_time: float = 10.0,
        max_captcha_wait: int = 120,
        proxy: Optional[str] = None,
        fingerprint_file: Optional[str] = None
    ):
        self.headless = headless
        self.slow_mo = slow_mo
        self.wait_time = wait_time
        self.max_captcha_wait = max_captcha_wait
        self.proxy = proxy
        
        # Load fingerprint
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
        
        self.ms_token: str = None
        self.cookies: Dict[str, str] = {}
        
        # Console log storage
        self.console_logs = []
        self.console_log_file = None
        
        # Network interception buffer
        self.intercepted_videos = []
        
        self.stats = {
            "videos_scraped": 0,
            "profiles_scraped": 0,
            "errors": 0,
            "api_calls": 0,
            "videos_from_hydration": 0,
            "videos_from_network": 0,
        }
    
    def _parse_proxy_url(self, proxy_url: str):
        """Parse proxy URL and extract credentials"""
        try:
            if proxy_url.count(":") == 3 and '@' not in proxy_url:
                ip, port, username, password = proxy_url.strip().split(":")
                self.proxy_username = username
                self.proxy_password = password
                self.proxy_server = f"http://{ip}:{port}"
                logger.info(f"Proxy server: {self.proxy_server}")
            else:
                parsed = urlparse(proxy_url if '://' in proxy_url else f'http://{proxy_url}')
                if parsed.username and parsed.password:
                    self.proxy_username = parsed.username
                    self.proxy_password = parsed.password
                if parsed.port:
                    self.proxy_server = f"{parsed.scheme or 'http'}://{parsed.hostname}:{parsed.port}"
                else:
                    self.proxy_server = f"{parsed.scheme or 'http'}://{parsed.hostname}:8080"
                logger.info(f"Proxy server: {self.proxy_server}")
        except Exception as e:
            logger.warning(f"Failed to parse proxy URL: {e}")
            self.proxy_server = proxy_url
    
    # ========================================================================
    # CONSOLE LOG CAPTURE
    # ========================================================================
    
    def _on_console(self, msg):
        """Capture browser console messages"""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            log_type = msg.type
            text = msg.text
            
            log_entry = {
                'timestamp': timestamp,
                'type': log_type,
                'text': text,
                'location': msg.location if hasattr(msg, 'location') else None
            }
            
            self.console_logs.append(log_entry)
            
            if log_type in ['error', 'warning']:
                logger.debug(f"[Browser {log_type.upper()}] {text}")
            elif 'stealth' in text.lower() or 'fingerprint' in text.lower():
                logger.info(f"[Browser Console] {text}")
            
            if self.console_log_file:
                self.console_log_file.write(f"[{timestamp}] [{log_type.upper()}] {text}\n")
                self.console_log_file.flush()
                
        except Exception as e:
            logger.debug(f"Error capturing console log: {e}")
    
    def start_console_logging(self, log_file: str = None):
        """Start capturing console logs to a file"""
        if not log_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = Path('logs/console')
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"console_{timestamp}.log"
        
        try:
            self.console_log_file = open(log_file, 'w', encoding='utf-8')
            logger.info(f"Console logging started: {log_file}")
            
            self.console_log_file.write("=" * 80 + "\n")
            self.console_log_file.write(f"Browser Console Log - {datetime.now().isoformat()}\n")
            self.console_log_file.write("=" * 80 + "\n\n")
            self.console_log_file.flush()
            
        except Exception as e:
            logger.error(f"Failed to open console log file: {e}")
            self.console_log_file = None
    
    def stop_console_logging(self):
        """Stop console logging and close file"""
        if self.console_log_file:
            try:
                self.console_log_file.write("\n" + "=" * 80 + "\n")
                self.console_log_file.write(f"Console logging ended - {datetime.now().isoformat()}\n")
                self.console_log_file.write("=" * 80 + "\n")
                self.console_log_file.close()
                logger.info("Console logging stopped")
            except Exception as e:
                logger.error(f"Error closing console log file: {e}")
            finally:
                self.console_log_file = None
    
    def save_console_logs_json(self, filename: str = None):
        """Save console logs as JSON for analysis"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = Path('logs/console')
            log_dir.mkdir(parents=True, exist_ok=True)
            filename = log_dir / f"console_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.console_logs, f, indent=2, ensure_ascii=False)
            logger.info(f"Console logs saved to JSON: {filename}")
        except Exception as e:
            logger.error(f"Failed to save console logs to JSON: {e}")
    
    # ========================================================================
    # BROWSER LIFECYCLE WITH BRAVE & STEALTH
    # ========================================================================
    
    async def start(self):
        """Start Brave browser with production-ready stealth configuration"""
        logger.info("=" * 70)
        logger.info("STARTING PRODUCTION-READY STEALTH BROWSER (BRAVE)")
        logger.info("=" * 70)
        
        self.playwright = await async_playwright().start()
        
        brave_path = find_brave()
        user_agent = self.fingerprint.get('userAgent', 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')
        
        launch_args = {
            'headless': self.headless,
            'slow_mo': self.slow_mo,
            'args': [
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-infobars',
                '--window-size=1920,1080',
                '--disable-extensions',
                '--disable-software-rasterizer',
                '--disable-gpu-sandbox',
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-background-networking',
            ],
            'ignore_default_args': ['--enable-automation'],
            'timeout': 60000
        }
        
        if brave_path:
            launch_args['executable_path'] = brave_path
            logger.info(f"✓ Using Brave: {brave_path}")
        else:
            logger.warning("⚠️ Brave not found, using Playwright's Chromium")
        
        if self.proxy_server:
            logger.info(f"Proxy: {self.proxy_server}")
            launch_args['proxy'] = {'server': self.proxy_server}
        elif self.proxy:
            launch_args['proxy'] = {'server': self.proxy}
        
        try:
            logger.info("Launching browser...")
            self.browser = await self.playwright.chromium.launch(**launch_args)
            logger.info("✓ Browser launched successfully")
        except Exception as e:
            logger.error(f"❌ Browser launch failed: {e}")
            logger.info("Trying fallback: Chromium without Brave...")
            
            if 'executable_path' in launch_args:
                del launch_args['executable_path']
            
            try:
                self.browser = await self.playwright.chromium.launch(**launch_args)
                logger.info("✓ Fallback successful: Using Chromium")
            except Exception as e2:
                logger.error(f"❌ Chromium also failed: {e2}")
                raise
        
        screen = self.fingerprint.get('screen', {'width': 1920, 'height': 1080})
        
        context_args = {
            'viewport': {'width': screen['width'], 'height': screen['height']},
            'user_agent': user_agent,
            'locale': 'en-US',
            'timezone_id': self.fingerprint.get('timezone', 'America/New_York'),
            'color_scheme': 'dark',
            'device_scale_factor': 1,
            'ignore_https_errors': True,
        }
        
        if self.proxy_server and self.proxy_username and self.proxy_password:
            context_args['http_credentials'] = {
                'username': self.proxy_username,
                'password': self.proxy_password,
            }
        
        try:
            logger.info("Creating browser context...")
            self.context = await self.browser.new_context(**context_args)
            logger.info("✓ Context created")
        except Exception as e:
            logger.error(f"❌ Context creation failed: {e}")
            raise
        
        await self.context.set_extra_http_headers({
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        })
        
        try:
            if os.path.exists('tiktok_cookies.json'):
                with open('tiktok_cookies.json', 'r') as f:
                    cookies = json.load(f)
                    sanitized_cookies = []
                    for cookie in cookies:
                        if 'sameSite' in cookie and cookie['sameSite'] not in ['Strict', 'Lax', 'None']:
                            del cookie['sameSite']
                        sanitized_cookies.append(cookie)
                    
                    await self.context.add_cookies(sanitized_cookies)
                    logger.info(f"✓ Loaded {len(sanitized_cookies)} cookies from tiktok_cookies.json")
            else:
                logger.warning("tiktok_cookies.json not found. You may need to log in.")
        except Exception as e:
            logger.error(f"Failed to load cookies: {e}")
        
        fingerprint_json = json.dumps(self.fingerprint)
        await self.context.add_init_script(f"""
            (() => {{
                'use strict';
                const FINGERPRINT = {fingerprint_json};
                
                if (FINGERPRINT.platform) {{
                    Object.defineProperty(Navigator.prototype, 'platform', {{
                        get: function() {{ return FINGERPRINT.platform; }},
                        configurable: true,
                        enumerable: true
                    }});
                }}
                
                if (FINGERPRINT.hardwareConcurrency) {{
                    Object.defineProperty(Navigator.prototype, 'hardwareConcurrency', {{
                        get: function() {{ return FINGERPRINT.hardwareConcurrency; }},
                        configurable: true,
                        enumerable: true
                    }});
                }}
                
                if (FINGERPRINT.deviceMemory) {{
                    Object.defineProperty(Navigator.prototype, 'deviceMemory', {{
                        get: function() {{ return FINGERPRINT.deviceMemory; }},
                        configurable: true,
                        enumerable: true
                    }});
                }}
                
                if (FINGERPRINT.languages) {{
                    Object.defineProperty(Navigator.prototype, 'languages', {{
                        get: function() {{ return FINGERPRINT.languages; }},
                        configurable: true,
                        enumerable: true
                    }});
                }}
                
                if (FINGERPRINT.maxTouchPoints !== undefined) {{
                    Object.defineProperty(Navigator.prototype, 'maxTouchPoints', {{
                        get: function() {{ return FINGERPRINT.maxTouchPoints; }},
                        configurable: true,
                        enumerable: true
                    }});
                }}
                
                const originalGetOwnPropertyNames = Object.getOwnPropertyNames;
                Object.getOwnPropertyNames = function(obj) {{
                    const props = originalGetOwnPropertyNames(obj);
                    if (obj === navigator) {{
                        return [];
                    }}
                    return props;
                }};
                
                const originalKeys = Object.keys;
                Object.keys = function(obj) {{
                    const keys = originalKeys(obj);
                    if (obj === navigator) {{
                        return [];
                    }}
                    return keys;
                }};
                
                const originalGetOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
                Object.getOwnPropertyDescriptor = function(obj, prop) {{
                    if (obj === navigator && prop === 'webdriver') {{
                        return undefined;
                    }}
                    return originalGetOwnPropertyDescriptor(obj, prop);
                }};
                
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
                
                if (typeof window.chrome === 'undefined') {{
                    window.chrome = {{
                        runtime: {{}},
                        app: {{}},
                        loadTimes: () => {{}},
                        csi: () => {{}}
                    }};
                }} else if (!window.chrome.runtime) {{
                    try {{
                        window.chrome.runtime = {{}};
                        window.chrome.app = window.chrome.app || {{}};
                        window.chrome.loadTimes = window.chrome.loadTimes || (() => {{}});
                        window.chrome.csi = window.chrome.csi || (() => {{}});
                    }} catch(e) {{}}
                }}
            }})();
        """)
        
        try:
            logger.info("Creating new page...")
            self.page = await self.context.new_page()
            logger.info("✓ Page created")
        except Exception as e:
            logger.error(f"❌ Page creation failed: {e}")
            raise
        
        self.page.on("console", self._on_console)
        self.start_console_logging()
        
        try:
            cdp = await self.context.new_cdp_session(self.page)
            await cdp.send('Page.addScriptToEvaluateOnNewDocument', {
                'source': """
                    delete Navigator.prototype.webdriver;
                """
            })
            logger.info("✓ CDP session established")
        except Exception as e:
            logger.warning(f"CDP session failed (not critical): {e}")
        
        logger.info("Navigating to TikTok...")
        await asyncio.sleep(random.uniform(1.0, 2.5))
        
        try:
            await self.page.goto('https://www.tiktok.com/', 
                                wait_until='domcontentloaded', 
                                timeout=45000)
            logger.info("✓ Successfully loaded TikTok")
        except Exception as e:
            logger.error(f"❌ Failed to load TikTok: {e}")
            try:
                screenshot_path = Path('logs') / f'error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                screenshot_path.parent.mkdir(exist_ok=True)
                await self.page.screenshot(path=str(screenshot_path))
                logger.info(f"Screenshot saved: {screenshot_path}")
            except:
                pass
            raise
        
        await asyncio.sleep(random.uniform(2.5, 4.0))
        await self._update_cookies()
        logger.info("✓ Browser ready!")
        logger.info("=" * 70)
    
    async def stop(self):
        """Stop the browser"""
        try:
            self.stop_console_logging()
            self.save_console_logs_json()
            
            if self.browser:
                await self.browser.close()
        except Exception as e:
            logger.debug(f"Error closing browser: {e}")
        
        try:
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            logger.debug(f"Error stopping playwright: {e}")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def _update_cookies(self):
        """Update cookies from browser"""
        browser_cookies = await self.context.cookies()
        for cookie in browser_cookies:
            self.cookies[cookie['name']] = cookie['value']
            if cookie['name'] == 'msToken':
                self.ms_token = cookie['value']
    
    async def _extract_page_data(self) -> Optional[Dict]:
        """Extract data from page scripts with proper serialization"""
        try:
            data = await self.page.evaluate('''
                () => {
                    function deepClone(obj) {
                        try {
                            return JSON.parse(JSON.stringify(obj));
                        } catch(e) {
                            return null;
                        }
                    }
                    
                    if (typeof window.__UNIVERSAL_DATA_FOR_REHYDRATION__ !== 'undefined') {
                        const data = window.__UNIVERSAL_DATA_FOR_REHYDRATION__;
                        const cloned = deepClone(data);
                        if (cloned) {
                            return cloned;
                        }
                    }
                    
                    if (typeof window.SIGI_STATE !== 'undefined') {
                        const data = window.SIGI_STATE;
                        const cloned = deepClone(data);
                        if (cloned) {
                            return cloned;
                        }
                    }
                    
                    return null;
                }
            ''')
            
            if data and isinstance(data, dict):
                logger.debug("✓ Page data extracted successfully")
                return data
            else:
                logger.debug("✗ No valid page data found")
                return None
            
        except Exception as e:
            logger.debug(f"Error extracting page data: {e}")
            return None
    
    # ========================================================================
    # VIDEO SCRAPING WITH NETWORK INTERCEPTION
    # ========================================================================
    
    async def get_user_videos(
        self,
        username: str,
        max_videos: int = 1000,
        milestone_datetime: Optional[datetime] = None
    ) -> List[VideoData]:
        """
        Get videos using BOTH initial hydration data AND network interception.
        This captures the first batch from page state and subsequent batches from API calls.
        """
        
        if milestone_datetime:
            cutoff_timestamp = int(milestone_datetime.timestamp())
            logger.info(f"Using milestone datetime: {milestone_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            cutoff_timestamp = 0
            logger.info("No milestone set - fetching all videos")
        
        url = f"https://www.tiktok.com/@{username}"
        videos = []
        seen_ids = set()
        
        self.intercepted_videos = []
        
        async def handle_api_response(response: Response):
            """Intercept and parse api/post/item_list responses"""
            try:
                if "api/post/item_list" in response.url and response.status == 200:
                    self.stats["api_calls"] += 1
                    logger.info(f"🌐 Intercepted API call: {response.url[:100]}...")
                    
                    try:
                        json_data = await response.json()
                        item_list = json_data.get("itemList", [])
                        
                        if item_list:
                            logger.info(f"✓ Found {len(item_list)} videos in API response")
                            self.intercepted_videos.extend(item_list)
                            self.stats["videos_from_network"] += len(item_list)
                        else:
                            logger.debug("API response had no itemList")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON from API response: {e}")
                    except Exception as e:
                        logger.error(f"Error parsing API response: {e}")
                        
            except Exception as e:
                logger.debug(f"Error in response handler: {e}")
        
        self.page.on("response", handle_api_response)
        
        try:
            logger.info(f"Navigating to @{username}...")
            await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            await asyncio.sleep(self.wait_time + 2)
            
            try:
                await self.page.wait_for_selector('[data-e2e="user-post-item"]', timeout=15000)
                logger.info("✓ Video grid loaded")
            except:
                logger.warning("Video grid not found - account might be private or empty")
                return []
            
            logger.info("📦 Extracting initial videos from page hydration data...")
            data = await self._extract_page_data()
            
            if data:
                initial_videos = self._extract_videos_from_page_data(data, username, cutoff_timestamp)
                for video in initial_videos:
                    if video.video_id not in seen_ids:
                        seen_ids.add(video.video_id)
                        videos.append(video)
                        self.stats["videos_scraped"] += 1
                        self.stats["videos_from_hydration"] += 1
                
                logger.info(f"✓ Extracted {len(initial_videos)} videos from hydration data")
            else:
                logger.warning("⚠️ Could not extract hydration data - will rely on network interception")
            
            scroll_count = 0
            max_scrolls = 50
            no_new_videos_count = 0
            reached_milestone = False
            
            logger.info("🔄 Starting scroll loop to trigger API calls...")
            
            while len(videos) < max_videos and scroll_count < max_scrolls and not reached_milestone:
                await self.page.evaluate('window.scrollBy(0, window.innerHeight * 2)')
                scroll_count += 1
                await asyncio.sleep(self.wait_time)
                
                if self.intercepted_videos:
                    videos_before = len(videos)
                    
                    for item in self.intercepted_videos:
                        video = self._parse_video_data(item, username)
                        
                        if video and video.video_id not in seen_ids:
                            if video.create_timestamp >= cutoff_timestamp:
                                seen_ids.add(video.video_id)
                                videos.append(video)
                                self.stats["videos_scraped"] += 1
                            else:
                                logger.info(f"⏹️ Reached milestone - video {video.video_id} is before cutoff")
                                reached_milestone = True
                                break
                    
                    new_count = len(videos) - videos_before
                    
                    if new_count > 0:
                        logger.info(f"Scroll {scroll_count}: +{new_count} videos (total: {len(videos)})")
                        no_new_videos_count = 0
                    else:
                        no_new_videos_count += 1
                    
                    self.intercepted_videos = []
                else:
                    no_new_videos_count += 1
                    logger.debug(f"Scroll {scroll_count}: No new videos intercepted")
                
                if no_new_videos_count >= 5:
                    logger.info("No new videos after 5 scrolls - stopping")
                    break
            
            logger.info(f"✅ Total videos fetched: {len(videos)}")
            logger.info(f"   - From hydration data: {self.stats['videos_from_hydration']}")
            logger.info(f"   - From network API: {self.stats['videos_from_network']}")
            logger.info(f"   - Total API calls: {self.stats['api_calls']}")
            
            return videos
            
        except Exception as e:
            logger.error(f"Error getting user videos: {e}")
            import traceback
            traceback.print_exc()
            return videos
            
        finally:
            try:
                self.page.remove_listener("response", handle_api_response)
            except:
                pass
    
    def _extract_videos_from_page_data(
        self, 
        data: Dict, 
        username: str, 
        cutoff_timestamp: int
    ) -> List[VideoData]:
        """Extract videos from initial page hydration data"""
        videos = []
        
        try:
            default_scope = data.get('__DEFAULT_SCOPE__', {})
            user_post = default_scope.get('webapp.user-post', {})
            item_list = user_post.get('itemList', [])
            
            if item_list and isinstance(item_list, list):
                item_list = [item for item in item_list if isinstance(item, dict)]
            
            if not item_list:
                item_module = data.get('ItemModule', {})
                if item_module and isinstance(item_module, dict):
                    item_list = [v for v in item_module.values() if isinstance(v, dict)]
            
            logger.debug(f"Found {len(item_list)} video objects in page data")
            
            for item in item_list:
                if not isinstance(item, dict):
                    continue
                    
                video = self._parse_video_data(item, username)
                if video:
                    if video.create_timestamp >= cutoff_timestamp:
                        videos.append(video)
                        logger.debug(f"Extracted video {video.video_id} from page data")
                    else:
                        logger.info(f"Found video before milestone in initial page data: {video.video_id}")
                        break
        
        except Exception as e:
            logger.debug(f"Error extracting videos from page data: {e}")
        
        return videos
    
    def _parse_video_data(self, item: Dict, username: str) -> Optional[VideoData]:
        """Parse video data from API response or hydration data"""
        try:
            if not isinstance(item, dict):
                return None
            
            if self._is_pinned_video(item):
                logger.debug(f"Skipping pinned video: {item.get('id')}")
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
    
    def _is_pinned_video(self, item: Dict) -> bool:
        """Check if a video is pinned to the profile"""
        try:
            if item.get('isPinnedItem') or item.get('pinned'):
                return True
            
            author = item.get('author', {})
            if isinstance(author, dict):
                pinned_item_ids = author.get('pinnedItemIds', [])
                video_id = str(item.get('id', ''))
                if video_id and video_id in pinned_item_ids:
                    return True
            
            if item.get('indexEnabled') is False:
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking pinned status: {e}")
            return False
    
    # ========================================================================
    # HIGH-LEVEL SCRAPING
    # ========================================================================
    
    async def scrape_user_profile(
        self,
        username: str,
        profile_dir: Path,
        max_videos: int = 1000,
        milestone_datetime: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Scrape user profile and save to profile directory"""
        logger.info(f"Starting scrape for @{username}")
        
        now = datetime.now()
        if milestone_datetime:
            cutoff_date = milestone_datetime
            logger.info(f"Using milestone: {milestone_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            cutoff_date = None
            logger.info("No milestone set - fetching all videos")
        
        if cutoff_date:
            logger.info(f"Date range: {cutoff_date.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}")
        
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        videos = await self.get_user_videos(username, max_videos, milestone_datetime)
        
        if not videos:
            logger.error("No videos fetched")
            return {"error": "No videos found"}
        
        logger.info(f"Fetched {len(videos)} videos")
        
        # Save raw videos
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        videos_file = profile_dir / f"videos_raw_{timestamp}.json"
        with open(videos_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(v) for v in videos], f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(videos)} videos to {videos_file}")
        
        # Generate summary
        summary = {
            "username": username,
            "milestone_datetime": milestone_datetime.strftime('%Y-%m-%d %H:%M:%S') if milestone_datetime else None,
            "date_range": {
                "start": cutoff_date.strftime('%Y-%m-%d') if cutoff_date else None,
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
                "videos": str(videos_file.name)
            }
        }
        
        # Save summary
        summary_file = profile_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved summary to {summary_file}")
        
        return summary


# ============================================================================
# MAIN EXECUTION (RUNNER)
# ============================================================================

async def main():
    # Configure Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("scraper.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # Configuration
    TARGET_USERNAMES = [
        "tiktok", 
        "khabylame",
    ]
    
    LOOKBACK_DAYS = 30 
    milestone = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    
    # Create run directory with timestamp
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path("video_data") / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"--- STARTING SCRAPE ---")
    logger.info(f"Milestone Date: {milestone.strftime('%Y-%m-%d %H:%M:%S')} ({LOOKBACK_DAYS} days ago)")
    logger.info(f"Output directory: {run_dir}")
    
    # Track run statistics
    run_stats = {
        "run_name": f"scrape_{run_timestamp}",
        "milestone_datetime": milestone.strftime('%Y-%m-%d %H:%M:%S'),
        "output_dir": str(run_dir),
        "started_at": datetime.now().isoformat(),
        "profiles_processed": 0,
        "profiles_success": 0,
        "profiles_failed": 0,
        "total_videos": 0,
        "profile_summaries": {}
    }

    async with PlaywrightScraper(headless=False) as scraper:
        
        for username in TARGET_USERNAMES:
            try:
                logger.info(f"Processing user: @{username}")
                run_stats["profiles_processed"] += 1
                
                # Create profile directory
                profile_dir = run_dir / username
                
                result = await scraper.scrape_user_profile(
                    username=username,
                    profile_dir=profile_dir,
                    max_videos=1000,
                    milestone_datetime=milestone
                )
                
                if "error" in result:
                    logger.error(f"Failed to scrape @{username}: {result['error']}")
                    run_stats["profiles_failed"] += 1
                else:
                    logger.info(f"✓ Successfully scraped @{username}")
                    logger.info(f"  - Videos captured: {result.get('total_videos')}")
                    logger.info(f"  - Output saved to: {profile_dir}/")
                    
                    run_stats["profiles_success"] += 1
                    run_stats["total_videos"] += result.get('total_videos', 0)
                    
                    # Add to profile summaries
                    run_stats["profile_summaries"][username] = {
                        "username": username,
                        "videos": result.get('total_videos', 0),
                        "date_range": result.get('date_range'),
                        "files": result.get('files')
                    }
                
                if username != TARGET_USERNAMES[-1]:
                    sleep_time = random.uniform(5, 10)
                    logger.info(f"Sleeping {sleep_time:.2f}s before next user...")
                    await asyncio.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Critical error processing {username}: {e}")
                run_stats["profiles_failed"] += 1
                import traceback
                traceback.print_exc()
    
    # Save run summary
    run_stats["completed_at"] = datetime.now().isoformat()
    run_stats["scraper_stats"] = scraper.stats
    
    run_summary_file = run_dir / f"run_summary_{run_timestamp}.json"
    with open(run_summary_file, 'w', encoding='utf-8') as f:
        json.dump(run_stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Run complete! Summary saved to {run_summary_file}")
    logger.info(f"   - Profiles processed: {run_stats['profiles_processed']}")
    logger.info(f"   - Success: {run_stats['profiles_success']}")
    logger.info(f"   - Failed: {run_stats['profiles_failed']}")
    logger.info(f"   - Total videos: {run_stats['total_videos']}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())