"""
Unified TikTok Profile Scraper with Two Pipelines

Pipeline 1: Fast metadata scraping - scrapes all video metadata from profiles
Pipeline 2: Deep detailed scraping - visits each video individually for comments + labels

All memory leaks fixed, production-ready
"""

import asyncio
import json
import logging
import os
import random
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict, field
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Page, BrowserContext, Browser, Response

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed. Install with: pip install psutil")

logger = logging.getLogger('TikTok.Scraper')


# ============================================================================
# BRAVE FINDER
# ============================================================================

def find_brave():
    """Find Brave Browser on the system."""
    paths = [
        '/usr/bin/brave-browser',
        '/usr/bin/brave',
        '/opt/brave.com/brave/brave',
        'C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe',
        '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser'
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
        try:
            with open(filepath, 'r') as f:
                fp = json.load(f)
                logger.info(f"✓ Loaded fingerprint from {filepath}")
                return fp
        except FileNotFoundError:
            logger.warning(f"Fingerprint file not found: {filepath}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid fingerprint JSON: {e}")
            return None
    
    @staticmethod
    def get_default_fingerprint() -> Dict:
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
# MEMORY MONITOR
# ============================================================================

class MemoryMonitor:
    """Monitor and log memory usage"""
    
    def __init__(self):
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        self.baseline_memory = None
        self.profile_memory_log = []
    
    def get_memory_mb(self) -> float:
        if self.process:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0
    
    def set_baseline(self):
        if self.process:
            self.baseline_memory = self.get_memory_mb()
            logger.info(f"📊 Baseline memory: {self.baseline_memory:.1f} MB")
    
    def log_memory(self, label: str = "Current"):
        if not self.process:
            return
        
        current = self.get_memory_mb()
        if self.baseline_memory:
            delta = current - self.baseline_memory
            logger.info(f"📊 {label} memory: {current:.1f} MB (Δ{delta:+.1f} MB)")
        else:
            logger.info(f"📊 {label} memory: {current:.1f} MB")
        
        return current
    
    def log_profile_memory(self, profile_num: int, username: str, before: float, after: float):
        delta = after - before
        entry = {
            "profile_num": profile_num,
            "username": username,
            "memory_before_mb": round(before, 1),
            "memory_after_mb": round(after, 1),
            "memory_delta_mb": round(delta, 1),
            "timestamp": datetime.now().isoformat()
        }
        self.profile_memory_log.append(entry)
        
        logger.info(f"📊 Profile #{profile_num} (@{username}): {before:.1f} → {after:.1f} MB (Δ{delta:+.1f} MB)")
        
        if delta > 100:
            logger.warning(f"⚠️ Large memory increase: +{delta:.1f} MB")
    
    def save_memory_log(self, filepath: Path):
        if not self.profile_memory_log:
            return
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "baseline_mb": self.baseline_memory,
                    "profiles": self.profile_memory_log
                }, f, indent=2)
            logger.info(f"Memory log saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save memory log: {e}")


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
    url: str
    diversification_labels: List[str] = field(default_factory=list)  # NEW: For Pipeline 2
    music_title: Optional[str] = None
    scraped_at: Optional[str] = None
    thumbnail_urls: Dict[str, str] = field(default_factory=dict)


@dataclass
class CommentData:
    """Data class for comment information (Pipeline 2)"""
    comment_id: str
    video_id: str
    text: str
    create_time: int
    create_time_iso: str
    digg_count: int
    reply_count: int
    user_id: str
    username: str
    nickname: str
    parent_comment_id: Optional[str] = None
    scraped_at: Optional[str] = None


# ============================================================================
# UNIFIED PLAYWRIGHT SCRAPER - BOTH PIPELINES
# ============================================================================

class PlaywrightScraper:
    """
    Unified TikTok scraper with two pipelines:
    - Pipeline 1: Fast metadata scraping
    - Pipeline 2: Deep scraping with comments and labels
    """
    
    def __init__(
        self,
        headless: bool = False,
        slow_mo: int = 50,
        wait_time: float = 5.0,
        proxy: Optional[str] = None,
        fingerprint_file: Optional[str] = None,
        restart_browser_every: int = 10,
        max_console_logs: int = 1000,
        memory_restart_mb: Optional[int] = None
    ):
        self.headless = headless
        self.slow_mo = slow_mo
        self.wait_time = wait_time
        self.restart_browser_every = restart_browser_every
        self.max_console_logs = max_console_logs
        self.memory_restart_mb = memory_restart_mb
        self._process = psutil.Process() if PSUTIL_AVAILABLE else None
        
        # Load fingerprint
        if fingerprint_file and Path(fingerprint_file).exists():
            self.fingerprint = BrowserFingerprint.load_from_file(fingerprint_file)
        else:
            if fingerprint_file:
                logger.warning(f"Fingerprint file not found: {fingerprint_file}, using default")
            self.fingerprint = BrowserFingerprint.get_default_fingerprint()
        
        # Parse proxy
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
        
        # Caches and buffers
        self.video_labels_cache: Dict[str, List[str]] = {}  # NEW: For label interception
        self.intercepted_videos = []
        self.intercepted_comments_buffer = []  # NEW: For comment interception
        self.console_logs = []
        self.console_log_file = None
        
        # Event handlers - CRITICAL FOR CLEANUP
        self._response_handler = None
        self._console_handler = None
        self._label_handler = None  # NEW: Label interceptor
        self._comment_handler = None  # NEW: Comment interceptor
        self._detail_handler = None  # NEW: Detail metadata interceptor
        
        self.stats = {
            "videos_scraped": 0,
            "profiles_scraped": 0,
            "comments_scraped": 0,  # NEW
            "errors": 0,
            "api_calls": 0,
            "videos_from_hydration": 0,
            "videos_from_network": 0,
            "browser_restarts": 0,
        }
    
    def _parse_proxy_url(self, proxy_url: str):
        """Parse proxy URL and extract credentials"""
        try:
            if proxy_url.count(":") == 3 and '@' not in proxy_url:
                ip, port, username, password = proxy_url.strip().split(":")
                self.proxy_username = username
                self.proxy_password = password
                self.proxy_server = f"http://{ip}:{port}"
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
    # CONSOLE LOGGING
    # ========================================================================
    
    def _on_console(self, msg):
        try:
            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            log_type = msg.type
            text = msg.text
            
            log_entry = {
                'timestamp': timestamp,
                'type': log_type,
                'text': text,
            }
            
            if len(self.console_logs) < self.max_console_logs:
                self.console_logs.append(log_entry)
            
            if self.console_log_file:
                self.console_log_file.write(f"[{timestamp}] [{log_type.upper()}] {text}\n")
                self.console_log_file.flush()
                
        except Exception as e:
            logger.debug(f"Error capturing console log: {e}")
    
    def clear_console_logs(self):
        self.console_logs.clear()
    
    def start_console_logging(self, log_file: str = None):
        if self.console_log_file:
            self.stop_console_logging()
        
        if not log_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = Path('logs/console')
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"console_{timestamp}.log"
        
        try:
            self.console_log_file = open(log_file, 'w', encoding='utf-8')
            self.console_log_file.write(f"Console Log - {datetime.now().isoformat()}\n\n")
            self.console_log_file.flush()
        except Exception as e:
            logger.error(f"Failed to open console log file: {e}")
            self.console_log_file = None
    
    def stop_console_logging(self):
        if self.console_log_file:
            try:
                self.console_log_file.close()
            except:
                pass
            finally:
                self.console_log_file = None
    
    # ========================================================================
    # BROWSER LIFECYCLE
    # ========================================================================
    
    async def start(self):
        """Start browser with stealth configuration"""
        logger.info("=" * 70)
        logger.info("STARTING BROWSER")
        logger.info("=" * 70)
        
        await self._ensure_clean_state()
        
        self.playwright = await async_playwright().start()
        
        brave_path = find_brave()
        user_agent = self.fingerprint.get('userAgent', 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
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
        
        if self.proxy_server:
            launch_args['proxy'] = {'server': self.proxy_server}
        
        try:
            self.browser = await self.playwright.chromium.launch(**launch_args)
            logger.info("✓ Browser launched")
        except Exception as e:
            logger.error(f"❌ Browser launch failed: {e}")
            if 'executable_path' in launch_args:
                del launch_args['executable_path']
            self.browser = await self.playwright.chromium.launch(**launch_args)
        
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
        
        
        # Add stealth scripts
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
        
        self.page = await self.context.new_page()
        
        # Setup console logging
        self._console_handler = self._on_console
        self.page.on("console", self._console_handler)
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
        """Stop browser and clean up all resources"""
        logger.debug("Stopping browser...")
        
        try:
            self.stop_console_logging()
            
            if self.page:
                # Remove all event listeners
                for handler in [self._response_handler, self._console_handler, 
                            self._label_handler, self._comment_handler]:
                    if handler:
                        try:
                            self.page.remove_listener("response", handler)
                        except:
                            pass
                
                await self.page.close()
            
            if self.context:
                await self.context.close()
            
            if self.browser:
                await self.browser.close()
        
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")
        
        finally:
            self.page = None
            self.context = None
            self.browser = None
            self._response_handler = None
            self._console_handler = None
            self._label_handler = None
            self._comment_handler = None
        
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        
        gc.collect()
    
    async def _ensure_clean_state(self):
        if self.browser or self.context or self.page:
            logger.warning("Existing browser detected, cleaning up...")
            await self.stop()
        
        self.intercepted_videos.clear()
        self.intercepted_comments_buffer.clear()
        self.video_labels_cache.clear()
        self.clear_console_logs()
        gc.collect()
    
    async def restart_browser(self):
        """Restart browser to prevent resource exhaustion"""
        logger.info("=" * 70)
        logger.info("♻️ RESTARTING BROWSER")
        logger.info("=" * 70)
        self.stats["browser_restarts"] += 1
        
        cookies_to_save = None
        try:
            if self.context:
                cookies_to_save = await self.context.cookies()
        except:
            pass
        
        await self.stop()
        await asyncio.sleep(3)
        gc.collect()
        
        await self.start()
        
        if cookies_to_save:
            try:
                await self.context.add_cookies(cookies_to_save)
            except:
                pass
        
        logger.info("✓ Browser restarted")
        logger.info("=" * 70)

    async def _maybe_restart_for_memory(self) -> bool:
        """Restart browser if RSS exceeds configured threshold."""
        if not self.memory_restart_mb or not self._process:
            return False
        try:
            rss_mb = self._process.memory_info().rss / 1024 / 1024
        except Exception:
            return False
        if rss_mb >= self.memory_restart_mb:
            logger.info(
                f"♻️ Restarting browser due to memory usage {rss_mb:.1f} MB >= {self.memory_restart_mb} MB"
            )
            await self.restart_browser()
            return True
        return False

    async def _update_cookies(self):
        """Update cookies from browser"""
        browser_cookies = await self.context.cookies()
        for cookie in browser_cookies:
            self.cookies[cookie['name']] = cookie['value']
            if cookie['name'] == 'msToken':
                self.ms_token = cookie['value']
    
    # ========================================================================
    # LABEL INTERCEPTION (Pipeline 2)
    # ========================================================================
    
    def _extract_labels_from_item(self, item: Dict):
        """Extract and cache diversification labels from video item"""
        if not isinstance(item, dict):
            return
        
        video_id = str(item.get('id', ''))
        if not video_id:
            return
        
        labels = item.get('diversificationLabels')
        
        if labels and isinstance(labels, list):
            self.video_labels_cache[video_id] = labels
            logger.debug(f"🏷️ Cached labels for video {video_id}: {labels}")
    
    async def _setup_label_interceptor(self):
        """Setup network interceptor to capture diversification labels"""
        
        async def handle_label_response(response: Response):
            target_endpoints = [
                "api/related/item_list",
                "api/item_detail",
                "api/post/item_list",
                "api/general/search"
            ]
            
            if response.status == 200 and any(ep in response.url for ep in target_endpoints):
                try:
                    json_data = await response.json()
                    
                    # Standard ItemList
                    item_list = json_data.get("itemList", [])
                    
                    # ItemStruct from detail API
                    if "itemInfo" in json_data:
                        item_struct = json_data["itemInfo"].get("itemStruct")
                        if item_struct:
                            item_list.append(item_struct)
                    
                    for item in item_list:
                        self._extract_labels_from_item(item)
                        
                except Exception as e:
                    logger.debug(f"Error in label interceptor: {e}")
        
        # Remove existing handler if any
        if self._label_handler:
            try:
                self.page.remove_listener("response", self._label_handler)
            except:
                pass
        
        self._label_handler = handle_label_response
        self.page.on("response", self._label_handler)
        logger.debug("✓ Label interceptor active")
    
    # ========================================================================
    # VIDEO PARSING (Both Pipelines)
    # ========================================================================
    
    async def _extract_page_data(self) -> Optional[Dict]:
        """Extract data from page scripts - extract specific paths to avoid circular refs"""
        try:
            # Extract just the parts we need to avoid circular reference issues
            data = await self.page.evaluate('''
                () => {
                    try {
                        console.log('[DEBUG] Starting extraction...');
                        
                        // Check both sources
                        const sources = {
                            UNIVERSAL: window.__UNIVERSAL_DATA_FOR_REHYDRATION__,
                            SIGI: window.SIGI_STATE
                        };
                        
                        let mainSource = null;
                        let sourceName = null;
                        
                        for (const [name, src] of Object.entries(sources)) {
                            if (src) {
                                console.log(`[DEBUG] ${name}: found (type: ${typeof src})`);
                                
                                // Try both Object.keys() and Object.getOwnPropertyNames()
                                const enumKeys = Object.keys(src);
                                const allKeys = Object.getOwnPropertyNames(src);
                                
                                console.log(`[DEBUG] ${name} enumerable keys:`, enumKeys.slice(0, 10));
                                console.log(`[DEBUG] ${name} all properties:`, allKeys.slice(0, 20));
                                
                                if (allKeys.length > 0) {
                                    mainSource = src;
                                    sourceName = name;
                                    break;
                                }
                            } else {
                                console.log(`[DEBUG] ${name}: not found`);
                            }
                        }
                        
                        if (!mainSource) {
                            console.log('[DEBUG] No valid source found');
                            return null;
                        }
                        
                        console.log(`[DEBUG] Using source: ${sourceName}`);
                        
                        // Extract only serializable parts
                        const result = {};
                        
                        // Get __DEFAULT_SCOPE__ if it exists
                        if (mainSource.__DEFAULT_SCOPE__) {
                            console.log('[DEBUG] Found __DEFAULT_SCOPE__');
                            result.__DEFAULT_SCOPE__ = {};
                            const scope = mainSource.__DEFAULT_SCOPE__;
                            
                            const scopeKeys = Object.getOwnPropertyNames(scope);
                            console.log('[DEBUG] __DEFAULT_SCOPE__ properties:', scopeKeys.slice(0, 20));
                            
                            // Extract video-detail
                            if (scope['webapp.video-detail']) {
                                try {
                                    result.__DEFAULT_SCOPE__['webapp.video-detail'] = 
                                        JSON.parse(JSON.stringify(scope['webapp.video-detail']));
                                    console.log('[DEBUG] ✓ Extracted webapp.video-detail');
                                } catch(e) {
                                    console.log('[DEBUG] ✗ Failed to extract webapp.video-detail:', e.message);
                                }
                            }
                            
                            // Extract item-detail
                            if (scope['webapp.item-detail']) {
                                try {
                                    result.__DEFAULT_SCOPE__['webapp.item-detail'] = 
                                        JSON.parse(JSON.stringify(scope['webapp.item-detail']));
                                    console.log('[DEBUG] ✓ Extracted webapp.item-detail');
                                } catch(e) {
                                    console.log('[DEBUG] ✗ Failed to extract webapp.item-detail:', e.message);
                                }
                            }
                        } else {
                            console.log('[DEBUG] __DEFAULT_SCOPE__ not found in source');
                        }
                        
                        // Get ItemModule if it exists
                        if (mainSource.ItemModule) {
                            console.log('[DEBUG] Found ItemModule');
                            try {
                                result.ItemModule = JSON.parse(JSON.stringify(mainSource.ItemModule));
                                console.log('[DEBUG] ✓ Extracted ItemModule');
                            } catch(e) {
                                console.log('[DEBUG] ✗ Failed to extract ItemModule:', e.message);
                            }
                        } else {
                            console.log('[DEBUG] ItemModule not found in source');
                        }
                        
                        const resultKeys = Object.keys(result);
                        console.log('[DEBUG] Final result keys:', resultKeys);
                        
                        if (resultKeys.length > 0) {
                            console.log('[DEBUG] ✓ Extraction successful');
                            return result;
                        }
                        
                        console.log('[DEBUG] No data could be extracted');
                        return null;
                    } catch(e) {
                        console.log('[DEBUG] ✗ Error in extraction:', e.message);
                        return null;
                    }
                }
            ''')
            
            if data and isinstance(data, dict):
                logger.debug(f"✓ Extracted page data with keys: {list(data.keys())}")
                return data
            
            logger.debug("Page data not found or serialization failed")
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting page data: {e}")
            return None
    
    def _parse_video_data(self, item: Dict, username: str) -> Optional[VideoData]:
        """Parse video data from API response (network-only)"""
        try:
            video_id = str(item.get('id', ''))
            if not video_id:
                return None

            author = item.get('author', {})
            if isinstance(author, str):
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

            url = f"https://www.tiktok.com/@{username}/video/{video_id}"
            
            # Extract diversification labels
            labels = item.get('diversificationLabels', []) or []

            video_dict = item.get('video', {})
            if not isinstance(video_dict, dict):
                video_dict = {}
            
            zoom_cover = video_dict.get('zoomCover', {})
            if not isinstance(zoom_cover, dict):
                zoom_cover = {}
            
            thumbnail_urls = {
                'origin': video_dict.get('originCover', ''),
                'cover': video_dict.get('cover', ''),
                'dynamic': video_dict.get('dynamicCover', ''),
                'zoom_960': zoom_cover.get('960', ''),
                'zoom_720': zoom_cover.get('720', ''),
                'zoom_480': zoom_cover.get('480', ''),
                'zoom_240': zoom_cover.get('240', ''),
            }
            
            return VideoData(
                video_id=video_id,
                author_username=username,
                author_id=str(author.get('id', '')),
                description=item.get('desc', ''),
                create_time=create_time_iso,
                create_timestamp=create_time,
                hashtags=hashtags,
                stats=normalized_stats,
                url=url,
                diversification_labels=labels,
                music_title=music.get('title', '') if music else None,
                scraped_at=datetime.now().isoformat(),
                thumbnail_urls=thumbnail_urls  # NEW
            )
        except Exception as e:
            logger.debug(f"Error parsing video: {e}")
            return None
    
    def _is_pinned_video(self, item: Dict) -> bool:
        """Check if video is pinned"""
        try:
            if item.get('isPinnedItem') or item.get('pinned'):
                return True
            
            author = item.get('author', {})
            if isinstance(author, dict):
                pinned_item_ids = author.get('pinnedItemIds', [])
                video_id = str(item.get('id', ''))
                if video_id and video_id in pinned_item_ids:
                    return True
            
            return False
            
        except:
            return False
    
    # ========================================================================
    # PIPELINE 1: FAST METADATA SCRAPING
    # ========================================================================
    
    async def get_user_videos(
        self,
        username: str,
        max_videos: int = 1000,
        milestone_datetime: Optional[datetime] = None
    ) -> List[VideoData]:
        """
        Get videos relying ONLY on network interception (api/post/item_list). No hydration parsing.
        """

        url = f"https://www.tiktok.com/@{username}"
        videos: List[VideoData] = []
        seen_ids: Set[str] = set()
        cutoff_timestamp = int(milestone_datetime.timestamp()) if milestone_datetime else 0

        self.intercepted_videos.clear()

        async def handle_api_response(response: Response):
            if "api/post/item_list" in response.url and response.status == 200:
                try:
                    json_data = await response.json()
                    item_list = json_data.get("itemList", [])
                    if item_list:
                        self.intercepted_videos.extend(item_list)
                        logger.info(f"🌐 Intercepted {len(item_list)} videos from API")
                except Exception:
                    pass

        self._response_handler = handle_api_response
        self.page.on("response", self._response_handler)

        try:
            logger.info(f"Navigating to @{username}...")
            await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            await asyncio.sleep(3)

            # Process any videos intercepted during initial page load (before scroll)
            if self.intercepted_videos:
                initial_batch = self.intercepted_videos.copy()
                self.intercepted_videos.clear()
                logger.info(f"🌐 Processing {len(initial_batch)} videos from initial page load")
                for item in initial_batch:
                    video = self._parse_video_data(item, username)
                    if video and video.video_id not in seen_ids:
                        if video.create_timestamp >= cutoff_timestamp:
                            seen_ids.add(video.video_id)
                            videos.append(video)

            logger.info(f"✓ Initial load complete: {len(videos)} videos captured")

            scroll_count = 0
            max_scrolls = 50
            no_new_data = 0
            reached_milestone = False

            while len(videos) < max_videos and scroll_count < max_scrolls:
                await self.page.evaluate('window.scrollBy(0, window.innerHeight * 2)')
                await asyncio.sleep(2)

                if self.intercepted_videos:
                    current_batch = self.intercepted_videos.copy()
                    self.intercepted_videos.clear()

                    added_in_batch = 0
                    for item in current_batch:
                        video = self._parse_video_data(item, username)
                        if video and video.video_id not in seen_ids:
                            if video.create_timestamp >= cutoff_timestamp:
                                seen_ids.add(video.video_id)
                                videos.append(video)
                                added_in_batch += 1
                            else:
                                reached_milestone = True

                    if reached_milestone:
                        logger.info("⏹️ Reached milestone date")
                        break

                    if added_in_batch > 0:
                        logger.info(f"Scroll {scroll_count}: +{added_in_batch} videos (Total: {len(videos)})")
                        no_new_data = 0
                    else:
                        no_new_data += 1
                else:
                    no_new_data += 1
                    logger.debug(f"Scroll {scroll_count}: No API data yet...")

                if no_new_data >= 5:
                    logger.info("No new API data after 5 scrolls. Stopping.")
                    break

                scroll_count += 1

            return videos

        finally:
            if self._response_handler:
                try:
                    self.page.remove_listener("response", self._response_handler)
                except Exception:
                    pass
                self._response_handler = None
            self.intercepted_videos.clear()
    
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
            
            for item in item_list:
                if not isinstance(item, dict):
                    continue
                    
                video = self._parse_video_data(item, username)
                if video:
                    if video.create_timestamp >= cutoff_timestamp:
                        videos.append(video)
                    else:
                        break
        
        except Exception as e:
            logger.debug(f"Error extracting videos from page data: {e}")
        
        return videos
    
    async def scrape_user_profile(
        self,
        username: str,
        profile_dir: Path,
        max_videos: int = 1000,
        milestone_datetime: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        PIPELINE 1: Fast profile scraping - metadata only
        """
        logger.info(f"🚀 PIPELINE 1: Fast scrape for @{username}")
        
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            videos = await self.get_user_videos(username, max_videos, milestone_datetime)
            
            if not videos:
                return {"error": "No videos found"}
            
            # Save raw videos
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            videos_file = profile_dir / f"videos_raw_{timestamp}.json"
            with open(videos_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(v) for v in videos], f, ensure_ascii=False, indent=2)
            
            # Generate summary
            summary = {
                "username": username,
                "scraped_at": datetime.now().isoformat(),
                "total_videos": len(videos),
                "total_stats": {
                    "total_views": sum(v.stats['playCount'] for v in videos),
                    "total_likes": sum(v.stats['diggCount'] for v in videos),
                    "total_comments": sum(v.stats['commentCount'] for v in videos),
                },
                "files": {
                    "videos": str(videos_file.name)
                }
            }
            
            summary_file = profile_dir / f"summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.stats["profiles_scraped"] += 1
            return summary
        
        finally:
            self.clear_console_logs()
    
    # ========================================================================
    # PIPELINE 2: DEEP SCRAPING WITH COMMENTS
    # ========================================================================
    
    async def get_video_comments(
        self, 
        video_id: str, 
        username: str, 
        max_comments: int = 40
    ) -> List[CommentData]:
        """
        Navigate to video and scrape comments using Mouse Wheel scrolling.
        """
        url = f"https://www.tiktok.com/@{username}/video/{video_id}"
        comments = []
        seen_ids = set()
        self.intercepted_comments_buffer.clear()
        
        async def handle_comment_response(response: Response):
            if "api/comment/list" in response.url and response.status == 200:
                try:
                    data = await response.json()
                    cms = data.get("comments", [])
                    if cms:
                        self.intercepted_comments_buffer.extend(cms)
                        logger.debug(f"📨 Intercepted {len(cms)} comments from API")
                except Exception:
                    pass
        
        self._comment_handler = handle_comment_response
        self.page.on("response", self._comment_handler)
        
        try:
            logger.debug(f"Navigating to video {video_id}...")
            await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            await asyncio.sleep(2)
            
            # NEW: Close any modals/popups that might be blocking
            await self._close_blocking_modals()
            
            # Click the comment button
            try:
                selector = '[data-e2e="comment-icon"]'
                await self.page.wait_for_selector(selector, timeout=5000)
                await self.page.click(selector)
                logger.info("✓ Clicked comment button")
                await asyncio.sleep(0.8)
            except:
                logger.warning("⚠️ Could not click comment button (might be already open or different layout)")

            # NEW: Close modals again after opening comments
            await self._close_blocking_modals()

            # Verify panel open
            comment_container_selector = 'div[class*="DivCommentListContainer"]'
            try:
                await self.page.wait_for_selector(comment_container_selector, timeout=5000)
                logger.info("✓ Comment panel verified open")
            except:
                logger.error("❌ Comment panel did not open")
                return []

            # Scroll loop using MOUSE WHEEL
            scrolls = 0
            max_scrolls = min(50, (max_comments // 20) + 5)
            no_new_data_count = 0
            
            logger.debug(f"Starting scroll loop (max {max_scrolls} scrolls)")
            
            while len(comments) < max_comments and scrolls < max_scrolls:
                try:
                    # Close any modals that might appear during scrolling
                    await self._close_blocking_modals()
                    
                    # Use Mouse Wheel without hover (which can be blocked by modals)
                    # Instead of hovering, just scroll on the page
                    await self.page.mouse.wheel(0, 3000)
                    await asyncio.sleep(random.uniform(0.8, 1.2))
                    
                    scrolls += 1
                    
                    # Process buffer
                    if self.intercepted_comments_buffer:
                        comments_before = len(comments)
                        while self.intercepted_comments_buffer:
                            c_raw = self.intercepted_comments_buffer.pop(0)
                            c_id = c_raw.get('cid')
                            if c_id and c_id not in seen_ids:
                                user = c_raw.get('user', {})
                                comments.append(CommentData(
                                    comment_id=c_id,
                                    video_id=video_id,
                                    text=c_raw.get('text', ''),
                                    create_time=int(c_raw.get('create_time', 0)),
                                    create_time_iso=datetime.fromtimestamp(int(c_raw.get('create_time', 0))).isoformat(),
                                    digg_count=c_raw.get('digg_count', 0),
                                    reply_count=c_raw.get('reply_comment_total', 0),
                                    user_id=user.get('uid', ''),
                                    username=user.get('unique_id', ''),
                                    nickname=user.get('nickname', ''),
                                    scraped_at=datetime.now().isoformat()
                                ))
                                seen_ids.add(c_id)
                                self.stats["comments_scraped"] += 1
                        
                        if len(comments) > comments_before:
                            logger.debug(f"Scroll {scrolls}: Total {len(comments)} comments")
                            no_new_data_count = 0
                        else:
                            no_new_data_count += 1
                    else:
                        no_new_data_count += 1

                    if no_new_data_count >= 5:
                        logger.debug("No new comments after 5 scrolls - stopping")
                        break

                except Exception as e:
                    logger.debug(f"Scroll error: {e}")
                    # Continue scrolling even if there's an error
                    await asyncio.sleep(1)

            return comments
            
        except Exception as e:
            logger.error(f"Error scraping comments: {e}")
            return comments
            
        finally:
            if self._comment_handler:
                try:
                    self.page.remove_listener("response", self._comment_handler)
                except:
                    pass
                self._comment_handler = None


    async def _close_blocking_modals(self):
        """
        Close any blocking modals/popups that prevent interaction
        """
        try:
            # Common modal close button selectors
            close_selectors = [
                'button[aria-label="Close"]',
                'button[data-e2e="modal-close-inner-button"]',
                'svg[data-e2e="modal-close-icon"]',
                '[class*="close-icon"]',
                '[class*="Close"]',
                'div[role="button"][aria-label="Close"]',
            ]
            
            for selector in close_selectors:
                try:
                    close_button = await self.page.query_selector(selector)
                    if close_button:
                        await close_button.click()
                        logger.debug(f"✓ Closed modal using {selector}")
                        await asyncio.sleep(0.5)
                        break
                except:
                    continue
            
            # Press Escape key as fallback
            try:
                await self.page.keyboard.press('Escape')
                await asyncio.sleep(0.3)
            except:
                pass
                
        except Exception as e:
            logger.debug(f"Modal close attempt: {e}")

    async def get_video_metadata_from_dom(self, page: Page) -> Optional[Dict]:
        """
        Extracts video metadata directly from the hydration script tag in the DOM.
        This bypasses network interception and window object availability issues.
        """
        try:
            logger.debug("Attempting to extract metadata from DOM script tag...")
            
            # 1. Target the script tag by the ID
            # We use state='attached' because the script is inside the DOM but not visible to the eye
            selector = '#__UNIVERSAL_DATA_FOR_REHYDRATION__'
            
            try:
                # Short timeout because if it's there, it's there immediately on load
                await page.wait_for_selector(selector, state='attached', timeout=2000)
            except:
                # Fallback: TikTok sometimes switches to this ID instead
                selector = '#SIGI_STATE'
                try:
                    await page.wait_for_selector(selector, state='attached', timeout=1000)
                except:
                    logger.debug("Hydration script tags not found in DOM.")
                    return None

            # 2. Extract the raw JSON string content
            json_text = await page.locator(selector).text_content()
            
            if not json_text:
                logger.warning("Found script tag but it was empty.")
                return None

            # 3. Parse JSON
            data = json.loads(json_text)
            item_struct = {}
            
            # 4. Navigate the JSON structure
            try:
                # Path A: Universal Data structure
                if '__DEFAULT_SCOPE__' in data:
                    default_scope = data['__DEFAULT_SCOPE__']
                    # Try video-detail first
                    video_detail = default_scope.get('webapp.video-detail', {})
                    item_struct = video_detail.get('itemInfo', {}).get('itemStruct', {})
                    
                    # Try item-detail if video-detail is empty
                    if not item_struct:
                        item_detail = default_scope.get('webapp.item-detail', {})
                        item_struct = item_detail.get('itemInfo', {}).get('itemStruct', {})

                # Path B: SIGI_STATE structure (usually keyed by Video ID)
                elif 'ItemModule' in data:
                    item_module = data['ItemModule']
                    # Get the first object key (Video ID)
                    first_key = next(iter(item_module))
                    item_struct = item_module[first_key]

                # 5. Extract the specific fields
                labels = item_struct.get('diversificationLabels')
                video_id = item_struct.get('id')
                
                if video_id:
                    logger.info(f"✅ Extracted metadata from DOM. Labels: {labels}")
                    
                    # Update your internal cache immediately
                    if labels:
                        self.video_labels_cache[video_id] = labels
                    
                    return item_struct
                else:
                    logger.warning("Parsed JSON but could not find valid itemStruct.")
                    return None

            except Exception as e:
                logger.error(f"Error navigating JSON structure: {e}")
                return None

        except Exception as e:
            logger.error(f"DOM extraction failed: {e}")
            return None

    async def get_video_metadata_from_network(self, url: str) -> Optional[VideoData]:
        """
        Navigate to video URL and capture metadata.
        Priority: 
        1. DOM Script Tag Extraction (Fastest, bypasses AdBlock/Network issues)
        2. API Interception (Fallback)
        """
        self.intercepted_video_detail = None
        self.captured_initial_data = None

        try:
            path_parts = urlparse(url).path.strip('/').split('/')
            username = path_parts[0].replace('@', '') if len(path_parts) > 0 else "unknown"
        except Exception:
            username = "unknown"

        # Setup API Listener (Fallback)
        async def detail_listener(response: Response):
            targets = ["api/item_detail", "api/iteminfo", "api/general/search/single", "api/item/detail"]
            if response.status == 200 and any(t in response.url for t in targets):
                try:
                    data = await response.json()
                    item = None
                    if isinstance(data, dict):
                        if data.get("itemInfo") and data["itemInfo"].get("itemStruct"):
                            item = data["itemInfo"]["itemStruct"]
                        elif data.get("itemStruct"):
                            item = data.get("itemStruct")
                    
                    if item:
                        self.intercepted_video_detail = item
                        logger.debug(f"🎯 Captured video metadata from API: {response.url[:80]}")
                except:
                    pass

        self._detail_handler = detail_listener
        self.page.on("response", self._detail_handler)

        try:
            logger.debug(f"Navigating to {url}")
            await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            
            # --- STRATEGY 1: DOM EXTRACTION (NEW FIX) ---
            # Try to grab data immediately from the HTML source
            dom_data = await self.get_video_metadata_from_dom(self.page)
            if dom_data:
                return self._parse_video_data(dom_data, username)
            
            # --- STRATEGY 2: NETWORK FALLBACK ---
            logger.debug("DOM extraction yielded no result, waiting for API response...")
            await asyncio.sleep(3) 

            # Check if we got data from API listener
            for _ in range(5):
                if self.intercepted_video_detail:
                    break
                await asyncio.sleep(0.5)

            if self.intercepted_video_detail:
                return self._parse_video_data(self.intercepted_video_detail, username)
            
            logger.warning("⚠️ Could not capture metadata from DOM or API")
            return None
            
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
            return None
        finally:
            if self._detail_handler:
                try:
                    self.page.remove_listener("response", self._detail_handler)
                except:
                    pass
                self._detail_handler = None

    async def run_pipeline_2a_fetch_list(self, username: str, lookback_days: int, run_timestamp: str):
        """Pipeline 2a: Save video URLs to txt using API interception only."""
        output_dir = Path("video_list") / run_timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "all_videos.txt"

        milestone = datetime.now() - timedelta(days=lookback_days)
        logger.info(f"📍 Fetching list for @{username} (Milestone: {milestone.date()})")

        videos = await self.get_user_videos(username, 1000, milestone)

        if videos:
            with open(output_file, 'a', encoding='utf-8') as f:
                for v in videos:
                    f.write(f"{v.url}\n")
            logger.info(f"✅ Saved {len(videos)} URLs to {output_file}")

            # Memory gate: restart if RSS exceeds threshold
            await self._maybe_restart_for_memory()
            return str(output_file)
        return None

    async def run_pipeline_2b_process_from_file(self, file_path: str, run_timestamp: str):
        """Pipeline 2b: Process video URL list from file (network-only metadata)."""
        if not os.path.exists(file_path):
            logger.error("File not found.")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]

        logger.info(f"📂 Processing {len(urls)} videos from {file_path}")

        input_path = Path(file_path)
        username_from_file = input_path.stem
        base_output_dir = Path("comments_data") / run_timestamp
        base_output_dir.mkdir(parents=True, exist_ok=True)

        await self._setup_label_interceptor()

        for idx, url in enumerate(urls, 1):
            try:
                logger.info(f"Processing {idx}/{len(urls)}: {url}")

                video_data = await self.get_video_metadata_from_network(url)
                if not video_data:
                    logger.warning("   ❌ Skipping (No metadata captured)")
                    continue

                if video_data.video_id in self.video_labels_cache:
                    video_data.diversification_labels = self.video_labels_cache[video_data.video_id]

                comments = await self.get_video_comments(video_data.video_id, video_data.author_username)
                logger.info(f"   ✓ Comments: {len(comments)} | Labels: {len(video_data.diversification_labels)}")

                data = {
                    "video_metadata": asdict(video_data),
                    "comments_count": len(comments),
                    "comments": [asdict(c) for c in comments]
                }

                user_dir = base_output_dir / (video_data.author_username or username_from_file)
                user_dir.mkdir(parents=True, exist_ok=True)
                save_path = user_dir / f"{video_data.video_id}.json"
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                # Memory gate: restart if RSS exceeds configured threshold
                if await self._maybe_restart_for_memory():
                    await self._setup_label_interceptor()
                    continue

                if idx % 5 == 0:
                    await asyncio.sleep(5)
                if idx % self.restart_browser_every == 0:
                    await self.restart_browser()
                    await self._setup_label_interceptor()

            except Exception as e:
                logger.error(f"Error on {url}: {e}")

    async def run_pipeline_2c_process_from_file(self, file_path: str, run_timestamp: str):
        """Pipeline 2c: Process video URL list from file (network-only metadata)."""
        if not os.path.exists(file_path):
            logger.error("File not found.")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]

        logger.info(f"📂 Processing {len(urls)} videos from {file_path}")

        input_path = Path(file_path)
        username_from_file = input_path.stem
        base_output_dir = Path("video_metadata") / run_timestamp
        base_output_dir.mkdir(parents=True, exist_ok=True)

        await self._setup_label_interceptor()

        for idx, url in enumerate(urls, 1):
            try:
                logger.info(f"Processing {idx}/{len(urls)}: {url}")

                video_data = await self.get_video_metadata_from_network(url)
                if not video_data:
                    logger.warning("   ❌ Skipping (No metadata captured)")
                    continue

                if video_data.video_id in self.video_labels_cache:
                    video_data.diversification_labels = self.video_labels_cache[video_data.video_id]

                data = {
                    "video_metadata": asdict(video_data),
                }

                user_dir = base_output_dir / (video_data.author_username or username_from_file)
                user_dir.mkdir(parents=True, exist_ok=True)
                save_path = user_dir / f"{video_data.video_id}.json"
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                # Memory gate: restart if RSS exceeds configured threshold
                if await self._maybe_restart_for_memory():
                    await self._setup_label_interceptor()
                    continue

                if idx % 5 == 0:
                    await asyncio.sleep(5)
                if idx % self.restart_browser_every == 0:
                    await self.restart_browser()
                    await self._setup_label_interceptor()

            except Exception as e:
                logger.error(f"Error on {url}: {e}")
    
    async def run_pipeline_2_detailed(
        self, 
        username: str, 
        lookback_days: int,
        run_timestamp: str,
        max_comments_per_video: int = 500,
        crawl_comments: bool = True
    ):
        """
        PIPELINE 2: Deep scraping with comments and labels
        
        1. Fetch recent videos (last N days)
        2. Visit each video individually
        3. Capture diversification labels via network interception
        4. Capture comments
        5. Save to: comments_data/{timestamp}/{username}/{video_id}.json
        """
        logger.info("=" * 70)
        logger.info(f"🚀 PIPELINE 2: Deep scrape for @{username} (Last {lookback_days} days)")
        logger.info("=" * 70)
        
        # Setup output directory
        base_dir = Path("comments_data") / run_timestamp / username
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Get video list (filtered by date)
        logger.info("Step 1: Fetching recent videos...")
        milestone = datetime.now() - timedelta(days=lookback_days)
        
        # Setup label interceptor for profile page
        await self._setup_label_interceptor()
        
        all_videos = await self.get_user_videos(username, max_videos=1000, milestone_datetime=milestone)
        
        # Filter by date
        cutoff_ts = int(milestone.timestamp())
        recent_videos = [v for v in all_videos if v.create_timestamp >= cutoff_ts]
        
        logger.info(f"✓ Found {len(recent_videos)} videos to process")
        
        # Step 2: Process each video
        logger.info("Step 2: Processing videos in detail...")
        
        for idx, video in enumerate(recent_videos, 1):
            logger.info(f"Processing {idx}/{len(recent_videos)}: Video {video.video_id}")
            
            try:
                # Ensure label interceptor is active
                await self._setup_label_interceptor()
                
                # Scrape comments (this also triggers label interception)
                if crawl_comments:
                    comments = await self.get_video_comments(
                        video.video_id, 
                        username, 
                        max_comments=max_comments_per_video
                    )
                
                # Check if we got labels from interception
                if not video.diversification_labels and video.video_id in self.video_labels_cache:
                    video.diversification_labels = self.video_labels_cache[video.video_id]
                    logger.info(f"   + Labels captured: {video.diversification_labels}")
                
                # Prepare output
                output_data = {
                    "video_metadata": asdict(video),
                    "comments_count": len(comments) if crawl_comments else 0,
                    "comments": [asdict(c) for c in comments] if crawl_comments else []
                }
                
                # Save immediately
                file_path = base_dir / f"{video.video_id}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"   ✓ Saved: {len(comments)} comments, Labels: {len(video.diversification_labels)}")
                
                # Rate limiting
                await asyncio.sleep(random.uniform(4, 8))
                
                # Memory management
                if idx % 10 == 0:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"   ❌ Error processing video {video.video_id}: {e}")
                continue
        
        logger.info("=" * 70)
        logger.info(f"✅ PIPELINE 2 COMPLETE: @{username}")
        logger.info(f"   - Videos processed: {len(recent_videos)}")
        logger.info(f"   - Output directory: {base_dir}")
        logger.info("=" * 70)
    
    # ========================================================================
    # PIPELINE 3: EXPLORE PAGE SCRAPING (RANDOM VIDEOS)
    # ========================================================================
    
    async def get_explore_videos(
        self,
        max_items: int = 200,
        lang: str = "vi",
        lookback_days: int = 0
    ) -> List[VideoData]:
        """
        Scrape videos from TikTok explore page filtered by language and date.
        
        Args:
            max_items: Maximum number of videos to collect
            lang: Filter by textLanguage (vi = Vietnamese)
            lookback_days: Limit to videos from last N days (0 = no limit)
        
        Returns:
            List of VideoData objects matching filters
        """
        url = "https://www.tiktok.com/explore"
        videos: List[VideoData] = []
        seen_ids: Set[str] = set()
        cutoff_timestamp = 0
        
        if lookback_days > 0:
            cutoff_timestamp = int((datetime.now() - timedelta(days=lookback_days)).timestamp())
        
        self.intercepted_videos.clear()

        async def handle_explore_response(response: Response):
            """Intercept explore/recommend API responses"""
            explore_endpoints = [
                "api/explore/item_list",
                "api/recommend/item_list",
                "api/general/search"
            ]
            
            if response.status == 200 and any(ep in response.url for ep in explore_endpoints):
                try:
                    json_data = await response.json()
                    item_list = json_data.get("itemList", [])
                    if item_list:
                        self.intercepted_videos.extend(item_list)
                        logger.info(f"🌐 Intercepted {len(item_list)} items from explore API")
                except Exception:
                    pass

        self._response_handler = handle_explore_response
        self.page.on("response", self._response_handler)

        try:
            logger.info(f"Navigating to explore page...")
            await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            await asyncio.sleep(3)

            # Process any videos intercepted during initial page load
            if self.intercepted_videos:
                initial_batch = self.intercepted_videos.copy()
                self.intercepted_videos.clear()
                logger.info(f"🌐 Processing {len(initial_batch)} items from initial page load")
                
                for item in initial_batch:
                    # Apply filters
                    if item.get("textLanguage") != lang:
                        continue
                    
                    # Check date filter if lookback_days is set
                    if lookback_days > 0:
                        create_timestamp = int(item.get('createTime', 0))
                        if create_timestamp < cutoff_timestamp:
                            continue
                    
                    # Extract username from author
                    author = item.get("author", {})
                    if isinstance(author, dict):
                        username = author.get("uniqueId", "explore")
                    else:
                        username = "explore"
                    
                    video = self._parse_video_data(item, username)
                    if video and video.video_id not in seen_ids:
                        seen_ids.add(video.video_id)
                        videos.append(video)

            logger.info(f"✓ Initial load complete: {len(videos)} videos captured (filtered)")

            # Scroll loop to load more videos
            scroll_count = 0
            no_new_data = 0

            while len(videos) < max_items:
                await self.page.evaluate('window.scrollBy(0, window.innerHeight * 2)')
                await asyncio.sleep(2)

                if self.intercepted_videos:
                    current_batch = self.intercepted_videos.copy()
                    self.intercepted_videos.clear()

                    added_in_batch = 0
                    for item in current_batch:
                        # Apply filters
                        if item.get("textLanguage") != lang:
                            continue
                        
                        # Check date filter if lookback_days is set
                        if lookback_days > 0:
                            create_timestamp = int(item.get('createTime', 0))
                            if create_timestamp < cutoff_timestamp:
                                continue
                        
                        # Extract username from author
                        author = item.get("author", {})
                        if isinstance(author, dict):
                            username = author.get("uniqueId", "explore")
                        else:
                            username = "explore"
                        
                        video = self._parse_video_data(item, username)
                        if video and video.video_id not in seen_ids:
                            seen_ids.add(video.video_id)
                            videos.append(video)
                            added_in_batch += 1

                    if added_in_batch > 0:
                        logger.info(f"Scroll {scroll_count}: +{added_in_batch} videos (Total: {len(videos)}/{max_items})")
                        no_new_data = 0
                    else:
                        no_new_data += 1
                else:
                    no_new_data += 1
                    logger.debug(f"Scroll {scroll_count}: No API data yet...")

                if no_new_data >= 5:
                    logger.info("No new API data after 5 scrolls. Stopping.")
                    break

                scroll_count += 1

            return videos

        finally:
            if self._response_handler:
                try:
                    self.page.remove_listener("response", self._response_handler)
                except Exception:
                    pass
                self._response_handler = None
            self.intercepted_videos.clear()
    
    async def run_pipeline_3_explore(
        self,
        num_runs: int = 5,
        max_items_per_run: int = 100,
        lang: str = "vi",
        lookback_days: int = 0,
        delay_between_runs: tuple = (5, 15)
    ) -> Dict[str, Any]:
        """
        PIPELINE 3: Explore page scraping - repeatedly crawl explore for random videos
        
        Crawls the explore page multiple times, filtering by language and date.
        Saves each run's results to separate files.
        
        Args:
            num_runs: Number of times to crawl explore page
            max_items_per_run: Max videos per run
            lang: Filter by textLanguage (default "vi" for Vietnamese)
            lookback_days: Limit to videos from last N days (0 = no limit)
            delay_between_runs: (min, max) seconds to wait between runs
        
        Returns:
            Summary dict with statistics
        """
        logger.info("=" * 70)
        logger.info(f"🚀 PIPELINE 3: Explore scraping ({num_runs} runs)")
        logger.info(f"   Language: {lang}")
        if lookback_days > 0:
            logger.info(f"   Lookback: {lookback_days} days")
        logger.info("=" * 70)
        
        # Setup output directory
        base_dir = Path("video_metadata/explore")
        base_dir.mkdir(parents=True, exist_ok=True)
        
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        all_videos = []
        total_unique = 0
        
        for run_num in range(1, num_runs + 1):
            logger.info(f"\n📍 Run {run_num}/{num_runs}")
            
            try:
                # Crawl explore page
                videos = await self.get_explore_videos(
                    max_items=max_items_per_run,
                    lang=lang,
                    lookback_days=lookback_days
                )
                
                if videos:
                    logger.info(f"   ✓ Collected {len(videos)} videos")
                    
                    # Save run results
                    run_file = base_dir / f"explore_{run_timestamp}_run{run_num:02d}.json"
                    with open(run_file, 'w', encoding='utf-8') as f:
                        json.dump([asdict(v) for v in videos], f, ensure_ascii=False, indent=2)
                    
                    # Track unique videos
                    for v in videos:
                        if v.video_id not in [existing.video_id for existing in all_videos]:
                            all_videos.append(v)
                    
                    total_unique = len(all_videos)
                    logger.info(f"   📊 Unique videos so far: {total_unique}")
                else:
                    logger.warning(f"   ⚠️ No videos collected in run {run_num}")
                
                # Delay between runs (except after last run)
                if run_num < num_runs:
                    delay = random.uniform(delay_between_runs[0], delay_between_runs[1])
                    logger.info(f"   ⏸️ Waiting {delay:.1f}s before next run...")
                    await asyncio.sleep(delay)
                
                # Memory management
                await self._maybe_restart_for_memory()
                
            except Exception as e:
                logger.error(f"   ❌ Error in run {run_num}: {e}")
                continue
        
        # Save consolidated results
        if all_videos:
            consolidated_file = base_dir / f"explore_{run_timestamp}_all.json"
            with open(consolidated_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(v) for v in all_videos], f, ensure_ascii=False, indent=2)
            
            logger.info(f"\n✅ Consolidated results: {len(all_videos)} unique videos")
            logger.info(f"   📁 Saved to: {consolidated_file}")
        
        # Generate summary
        summary = {
            "pipeline": 3,
            "run_timestamp": run_timestamp,
            "total_runs": num_runs,
            "videos_per_run": max_items_per_run,
            "language": lang,
            "total_videos_collected": len(all_videos),
            "unique_videos": total_unique,
            "output_directory": str(base_dir),
            "scraped_at": datetime.now().isoformat()
        }
        
        summary_file = base_dir / f"explore_{run_timestamp}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info("=" * 70)
        logger.info(f"✅ PIPELINE 3 COMPLETE")
        logger.info(f"   - Total runs: {num_runs}")
        logger.info(f"   - Total unique videos: {total_unique}")
        logger.info(f"   - Output directory: {base_dir}")
        logger.info("=" * 70)
        
        return summary

    # ============================================================================
    # PIPELINE 4: BULK THUMBNAL SCRAPING
    # ============================================================================

    async def download_thumbnail(
        self, 
        video_id: str, 
        thumbnail_url: str, 
        output_path: Path,
        max_retries: int = 3
    ) -> bool:
        """
        Download thumbnail using CDP fetch (properly handles proxy)
        
        Returns:
            True if successful, False otherwise
        """
        if not thumbnail_url:
            logger.debug(f"No thumbnail URL for {video_id}")
            return False
        
        for attempt in range(max_retries):
            try:
                # Use page.evaluate to fetch from browser context
                # This properly uses the browser's proxy settings
                result = await self.page.evaluate("""
                    async (url) => {
                        try {
                            const response = await fetch(url, {
                                headers: {
                                    'Referer': 'https://www.tiktok.com/'
                                }
                            });
                            
                            if (!response.ok) {
                                return { success: false, status: response.status };
                            }
                            
                            const blob = await response.blob();
                            const reader = new FileReader();
                            
                            return new Promise((resolve) => {
                                reader.onloadend = () => {
                                    const base64 = reader.result.split(',')[1];
                                    resolve({ success: true, data: base64 });
                                };
                                reader.readAsDataURL(blob);
                            });
                        } catch (error) {
                            return { success: false, error: error.message };
                        }
                    }
                """, thumbnail_url)
                
                if result.get('success'):
                    # Decode base64 and save
                    import base64
                    image_data = base64.b64decode(result['data'])
                    
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'wb') as f:
                        f.write(image_data)
                    
                    logger.debug(f"✓ Downloaded thumbnail: {output_path.name}")
                    return True
                else:
                    status = result.get('status', 'unknown')
                    error = result.get('error', 'unknown error')
                    logger.warning(f"Failed to download (attempt {attempt + 1}): {status} - {error}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        
            except Exception as e:
                logger.error(f"Error downloading thumbnail (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        
        return False

    async def run_pipeline_4_thumbnails(
        self,
        usernames: List[str],
        lookback_days: int,
        run_timestamp: str,
        quality: str = '960'  # Options: '960', '720', '480', '240', 'origin'
    ):
        """
        PIPELINE 4: Bulk thumbnail fetching
        
        1. Fetch video list for each user (like Pipeline 2a)
        2. Extract thumbnail URLs from intercepted data
        3. Download all thumbnails
        
        Args:
            usernames: List of TikTok usernames
            lookback_days: Only fetch videos from last N days
            run_timestamp: Timestamp for organizing output
            quality: Thumbnail quality ('960', '720', '480', '240', 'origin')
        """
        logger.info("=" * 70)
        logger.info(f"🚀 PIPELINE 4: Bulk Thumbnail Fetching ({quality}p)")
        logger.info("=" * 70)
        logger.info(f"Users: {len(usernames)}")
        logger.info(f"Lookback: {lookback_days} days")
        
        # Setup single output directory for all thumbnails
        thumbnails_dir = Path("thumbnails") / run_timestamp
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        
        milestone = datetime.now() - timedelta(days=lookback_days)
        
        # Statistics
        total_videos = 0
        total_downloaded = 0
        total_failed = 0
        all_metadata = []
        
        for idx, username in enumerate(usernames, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"USER {idx}/{len(usernames)}: @{username}")
            logger.info(f"{'='*70}")
            
            try:
                # Step 1: Fetch video metadata (reusing existing method)
                logger.info("Fetching video list...")
                videos = await self.get_user_videos(username, max_videos=1000, milestone_datetime=milestone)
                
                if not videos:
                    logger.warning(f"No videos found for @{username}")
                    continue
                
                logger.info(f"✓ Found {len(videos)} videos")
                total_videos += len(videos)
                
                # Step 3: Download thumbnails
                logger.info(f"Downloading {quality}p thumbnails...")
                
                downloaded = 0
                failed = 0
                
                for video_idx, video in enumerate(videos, 1):
                    # Select thumbnail URL based on quality preference
                    if quality == 'origin':
                        thumbnail_url = video.thumbnail_urls.get('origin')
                    else:
                        thumbnail_url = video.thumbnail_urls.get(f'zoom_{quality}')
                    
                    # Fallback to other qualities if preferred not available
                    if not thumbnail_url:
                        for fallback in ['zoom_960', 'zoom_720', 'zoom_480', 'origin', 'cover']:
                            thumbnail_url = video.thumbnail_urls.get(fallback)
                            if thumbnail_url:
                                logger.debug(f"Using fallback quality: {fallback}")
                                break
                    
                    if not thumbnail_url:
                        logger.warning(f"No thumbnail URL found for video {video.video_id}")
                        failed += 1
                        continue
                    
                    # Download with new naming: cover_{video_id}.jpg
                    output_path = thumbnails_dir / f"cover_{video.video_id}.jpg"
                    
                    # Skip if already exists
                    if output_path.exists():
                        logger.debug(f"Skipping {video.video_id} (already exists)")
                        downloaded += 1
                        
                        # Still add to metadata
                        all_metadata.append({
                            "video_id": video.video_id,
                            "username": username,
                            "url": video.url,
                            "thumbnail_local": str(output_path),
                            "thumbnail_urls": video.thumbnail_urls,
                            "create_time": video.create_time,
                            "stats": video.stats,
                            "hashtags": video.hashtags,
                            "description": video.description
                        })
                        continue
                    
                    success = await self.download_thumbnail(
                        video.video_id,
                        thumbnail_url,
                        output_path
                    )
                    
                    if success:
                        downloaded += 1
                        # Add to metadata
                        all_metadata.append({
                            "video_id": video.video_id,
                            "username": username,
                            "url": video.url,
                            "thumbnail_local": str(output_path),
                            "thumbnail_urls": video.thumbnail_urls,
                            "create_time": video.create_time,
                            "stats": video.stats,
                            "hashtags": video.hashtags,
                            "description": video.description
                        })
                    else:
                        failed += 1
                    
                    # Progress update every 10 videos
                    if video_idx % 10 == 0:
                        logger.info(f"Progress: {video_idx}/{len(videos)} ({downloaded} downloaded, {failed} failed)")
                    
                    # Rate limiting
                    if video_idx % 5 == 0:
                        await asyncio.sleep(random.uniform(0.5, 1.5))
                
                # User summary
                logger.info(f"✅ @{username} complete: {downloaded}/{len(videos)} downloaded, {failed} failed")
                total_downloaded += downloaded
                total_failed += failed
                
                # Memory management
                await self._maybe_restart_for_memory()
                
                # Delay between users
                if idx < len(usernames):
                    delay = random.uniform(5, 10)
                    logger.info(f"Waiting {delay:.1f}s before next user...")
                    await asyncio.sleep(delay)
            
            except Exception as e:
                logger.error(f"Error processing @{username}: {e}")
                continue
        
        # Save consolidated metadata for all videos
        metadata_file = thumbnails_dir / f"metadata_all_{run_timestamp}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved consolidated metadata: {metadata_file}")
        
        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("✅ PIPELINE 4 COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total users processed: {len(usernames)}")
        logger.info(f"Total videos found: {total_videos}")
        logger.info(f"Total thumbnails downloaded: {total_downloaded}")
        logger.info(f"Total failed: {total_failed}")
        logger.info(f"Success rate: {(total_downloaded/total_videos*100):.1f}%")
        logger.info(f"Output directory: {thumbnails_dir}")
        logger.info("=" * 70)
        
        # Save final summary
        summary = {
            "pipeline": 4,
            "run_timestamp": run_timestamp,
            "quality": quality,
            "lookback_days": lookback_days,
            "total_users": len(usernames),
            "total_videos": total_videos,
            "total_downloaded": total_downloaded,
            "total_failed": total_failed,
            "success_rate": round(total_downloaded/total_videos*100, 2) if total_videos > 0 else 0,
            "output_directory": str(thumbnails_dir),
            "scraped_at": datetime.now().isoformat()
        }
        
        summary_file = thumbnails_dir / f"summary_{run_timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def run_pipeline_1(usernames: List[str], lookback_days: int = 30):
    """Execute Pipeline 1: Fast metadata scraping"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("pipeline1_scraper.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    milestone = datetime.now() - timedelta(days=lookback_days)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path("video_data") / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("PIPELINE 1: FAST METADATA SCRAPING")
    logger.info("=" * 70)
    logger.info(f"Profiles: {len(usernames)}")
    logger.info(f"Milestone: {milestone.strftime('%Y-%m-%d')}")
    
    memory_monitor = MemoryMonitor()
    memory_monitor.set_baseline()
    
    scraper = PlaywrightScraper(
        headless=False,
        restart_browser_every=10
    )
    
    try:
        await scraper.start()
        
        for idx, username in enumerate(usernames):
            profile_num = idx + 1
            
            if profile_num > 1 and profile_num % scraper.restart_browser_every == 0:
                await scraper.restart_browser()
            
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"PROFILE {profile_num}/{len(usernames)}: @{username}")
                logger.info(f"{'='*70}")
                
                mem_before = memory_monitor.get_memory_mb()
                
                profile_dir = run_dir / username
                result = await scraper.scrape_user_profile(
                    username=username,
                    profile_dir=profile_dir,
                    max_videos=1000,
                    milestone_datetime=milestone
                )
                
                mem_after = memory_monitor.get_memory_mb()
                memory_monitor.log_profile_memory(profile_num, username, mem_before, mem_after)
                
                if "error" not in result:
                    logger.info(f"✅ Success: {result.get('total_videos')} videos")
                else:
                    logger.error(f"❌ Failed: {result.get('error')}")
                
                if username != usernames[-1]:
                    await asyncio.sleep(random.uniform(10, 20))
                    
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                
    finally:
        await scraper.stop()
        memory_monitor.log_memory("Final")
        
        # Save memory log
        memory_log_file = run_dir / f"memory_log_{run_timestamp}.json"
        memory_monitor.save_memory_log(memory_log_file)


async def run_pipeline_2(usernames: List[str], lookback_days: int = 30):
    """Execute Pipeline 2: Deep scraping with comments"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("pipeline2_scraper.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    logger.info("=" * 70)
    logger.info("PIPELINE 2: DEEP SCRAPING WITH COMMENTS")
    logger.info("=" * 70)
    logger.info(f"Profiles: {len(usernames)}")
    logger.info(f"Lookback: {lookback_days} days")
    logger.info(f"Run timestamp: {run_timestamp}")
    
    scraper = PlaywrightScraper(
        headless=False,
        restart_browser_every=5  # More aggressive for Pipeline 2
    )
    
    try:
        await scraper.start()
        
        for idx, username in enumerate(usernames):
            try:
                await scraper.run_pipeline_2_detailed(username, lookback_days, run_timestamp)
                
                # Restart between users
                if username != usernames[-1]:
                    await scraper.restart_browser()
                    await asyncio.sleep(10)
                    
            except Exception as e:
                logger.error(f"Failed to process {username}: {e}")
                
    finally:
        await scraper.stop()

async def run_pipeline_4(usernames: List[str], lookback_days: int = 30, quality: str = '960'):
    """Execute Pipeline 4: Bulk thumbnail fetching"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("pipeline4_thumbnails.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    logger.info("=" * 70)
    logger.info("PIPELINE 4: BULK THUMBNAIL FETCHING")
    logger.info("=" * 70)
    logger.info(f"Profiles: {len(usernames)}")
    logger.info(f"Quality: {quality}p")
    logger.info(f"Lookback: {lookback_days} days")
    logger.info(f"Run timestamp: {run_timestamp}")
    
    scraper = PlaywrightScraper(
        headless=False,
        restart_browser_every=10
    )
    
    try:
        await scraper.start()
        
        summary = await scraper.run_pipeline_4_thumbnails(
            usernames=usernames,
            lookback_days=lookback_days,
            run_timestamp=run_timestamp,
            quality=quality
        )
        
        logger.info("\n" + "="*70)
        logger.info("FINAL SUMMARY")
        logger.info("="*70)
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
        
    finally:
        await scraper.stop()

async def main():
    """Main entry point - choose your pipeline"""
    
    # Configuration
    TARGET_USERNAMES = [
        "tiktok",
        "khabylame",
        # Add more usernames...
    ]
    
    LOOKBACK_DAYS = 30
    
    # Choose pipeline:
    PIPELINE = 1  # Set to 1 or 2
    
    if PIPELINE == 1:
        await run_pipeline_1(TARGET_USERNAMES, LOOKBACK_DAYS)
    elif PIPELINE == 2:
        await run_pipeline_2(TARGET_USERNAMES, LOOKBACK_DAYS)
    elif PIPELINE == 3:
        # Pipeline 3 requires different parameters
        scraper = PlaywrightScraper(headless=False)
        await scraper.start()
        await scraper.run_pipeline_3_explore(
            num_runs=5,
            max_items_per_run=100,
            lang="vi",
            lookback_days=LOOKBACK_DAYS
        )
        await scraper.stop()
    elif PIPELINE == 4:
        await run_pipeline_4(TARGET_USERNAMES, LOOKBACK_DAYS, THUMBNAIL_QUALITY)
    else:
        print("Invalid pipeline selection. Choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())