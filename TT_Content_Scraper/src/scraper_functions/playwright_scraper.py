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
        max_console_logs: int = 1000
    ):
        self.headless = headless
        self.slow_mo = slow_mo
        self.wait_time = wait_time
        self.restart_browser_every = restart_browser_every
        self.max_console_logs = max_console_logs
        
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
                scraped_at=datetime.now().isoformat()
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
        max_comments: int = 500
    ) -> List[CommentData]:
        """
        Navigate to video and scrape comments
        ✅ Verifies comment panel opens before scraping
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
                except Exception as e:
                    logger.debug(f"Error parsing comment API response: {e}")
        
        self._comment_handler = handle_comment_response
        self.page.on("response", self._comment_handler)
        
        try:
            logger.debug(f"Navigating to video {video_id}...")
            await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            await asyncio.sleep(3)
            
            # Click the comment button to open comments panel
            comment_button_selectors = [
                '[data-e2e="comment-icon"]',
                'button[aria-label*="comment"]',
                'button[aria-label*="Comment"]',
                '[data-e2e="browse-comment"]',
            ]
            
            comment_button_clicked = False
            for selector in comment_button_selectors:
                try:
                    logger.debug(f"Trying to click comment button: {selector}")
                    await self.page.wait_for_selector(selector, timeout=5000)
                    await self.page.click(selector)
                    logger.info("✓ Clicked comment button")
                    comment_button_clicked = True
                    await asyncio.sleep(2)
                    break
                except Exception as e:
                    logger.debug(f"Failed to click {selector}: {e}")
                    continue
            
            if not comment_button_clicked:
                logger.warning("⚠️ Could not find/click comment button")
            
            # Verify comment panel opened
            try:
                await self.page.wait_for_selector(
                    '[class*="DivCommentListContainer"]',
                    timeout=5000,
                    state='visible'
                )
                logger.info("✓ Comment panel verified open")
            except Exception as e:
                logger.error(f"❌ Comment panel did not open: {e}")
                return []
            
            # Wait for initial comments to load
            await asyncio.sleep(2)
            
            # Process any already-intercepted comments
            if self.intercepted_comments_buffer:
                logger.debug(f"Processing {len(self.intercepted_comments_buffer)} initial comments")
            
            # Scroll to load more comments
            scrolls = 0
            max_scrolls = min(30, (max_comments // 20) + 5)
            no_new_comments_count = 0
            
            logger.debug(f"Starting scroll loop (max {max_scrolls} scrolls)")
            
            while len(comments) < max_comments and scrolls < max_scrolls:
                # Scroll the comment list container
                try:
                    scroll_result = await self.page.evaluate('''
                        () => {
                            // Try to find scrollable comment container
                            const selectors = [
                                '[class*="DivCommentListContainer"]',
                                '[class*="CommentListContainer"]',
                                '[data-e2e="comment-list"]',
                                '[class*="comment-list"]',
                                'div[class*="DivContainer"][style*="overflow"]'
                            ];
                            
                            let scrolled = false;
                            let scrolledSelector = null;
                            let oldScrollTop = 0;
                            let newScrollTop = 0;
                            
                            for (const selector of selectors) {
                                const container = document.querySelector(selector);
                                if (container) {
                                    const hasScroll = container.scrollHeight > container.clientHeight;
                                    const isScrollable = window.getComputedStyle(container).overflowY !== 'visible';
                                    
                                    if (hasScroll && isScrollable) {
                                        oldScrollTop = container.scrollTop;
                                        container.scrollTop = container.scrollTop + 800;
                                        newScrollTop = container.scrollTop;
                                        
                                        if (newScrollTop > oldScrollTop) {
                                            scrolled = true;
                                            scrolledSelector = selector;
                                            break;
                                        }
                                    }
                                }
                            }
                            
                            return {
                                scrolled: scrolled,
                                selector: scrolledSelector,
                                scrollAmount: newScrollTop - oldScrollTop
                            };
                        }
                    ''')
                    
                    if scroll_result['scrolled']:
                        logger.debug(f"✓ Scrolled {scroll_result['scrollAmount']}px in {scroll_result['selector']}")
                    else:
                        logger.debug("⚠️ Could not find scrollable container, trying page scroll")
                        await self.page.evaluate('window.scrollBy(0, 800)')
                        
                except Exception as e:
                    logger.debug(f"Scroll error: {e}")
                    # Fallback
                    await self.page.evaluate('window.scrollBy(0, 800)')
                
                # Wait for new comments to load
                await asyncio.sleep(2.5)
                scrolls += 1
                
                # Process intercepted comments
                if self.intercepted_comments_buffer:
                    comments_before = len(comments)
                    
                    while self.intercepted_comments_buffer:
                        c_raw = self.intercepted_comments_buffer.pop(0)
                        
                        try:
                            c_id = c_raw.get('cid')
                            if c_id and c_id not in seen_ids:
                                user = c_raw.get('user', {})
                                comments.append(CommentData(
                                    comment_id=c_id,
                                    video_id=video_id,
                                    text=c_raw.get('text', ''),
                                    create_time=int(c_raw.get('create_time', 0)),
                                    create_time_iso=datetime.fromtimestamp(
                                        int(c_raw.get('create_time', 0))
                                    ).isoformat(),
                                    digg_count=c_raw.get('digg_count', 0),
                                    reply_count=c_raw.get('reply_comment_total', 0),
                                    user_id=user.get('uid', ''),
                                    username=user.get('unique_id', ''),
                                    nickname=user.get('nickname', ''),
                                    scraped_at=datetime.now().isoformat()
                                ))
                                seen_ids.add(c_id)
                                self.stats["comments_scraped"] += 1
                        except Exception as e:
                            logger.debug(f"Error parsing comment: {e}")
                    
                    new_comments = len(comments) - comments_before
                    if new_comments > 0:
                        logger.debug(f"Scroll {scrolls}: +{new_comments} comments (total: {len(comments)})")
                        no_new_comments_count = 0
                    else:
                        no_new_comments_count += 1
                else:
                    no_new_comments_count += 1
                
                # Stop if no new comments after 4 scrolls (increased from 3)
                if no_new_comments_count >= 4:
                    logger.debug("No new comments after 4 scrolls - stopping")
                    break
                
                # Check if we've reached the bottom
                if scrolls > 5:  # Only check after a few scrolls
                    try:
                        at_bottom = await self.page.evaluate('''
                            () => {
                                const selectors = [
                                    '[class*="DivCommentListContainer"]',
                                    '[class*="CommentListContainer"]',
                                    '[data-e2e="comment-list"]'
                                ];
                                
                                for (const selector of selectors) {
                                    const container = document.querySelector(selector);
                                    if (container) {
                                        const scrollDiff = container.scrollHeight - container.scrollTop - container.clientHeight;
                                        return scrollDiff < 50;  // Within 50px of bottom
                                    }
                                }
                                return false;
                            }
                        ''')
                        
                        if at_bottom and no_new_comments_count >= 2:
                            logger.debug("Reached container bottom")
                            break
                    except Exception:
                        pass
            
            # Process any remaining comments in buffer
            while self.intercepted_comments_buffer:
                c_raw = self.intercepted_comments_buffer.pop(0)
                try:
                    c_id = c_raw.get('cid')
                    if c_id and c_id not in seen_ids:
                        user = c_raw.get('user', {})
                        comments.append(CommentData(
                            comment_id=c_id,
                            video_id=video_id,
                            text=c_raw.get('text', ''),
                            create_time=int(c_raw.get('create_time', 0)),
                            create_time_iso=datetime.fromtimestamp(
                                int(c_raw.get('create_time', 0))
                            ).isoformat(),
                            digg_count=c_raw.get('digg_count', 0),
                            reply_count=c_raw.get('reply_comment_total', 0),
                            user_id=user.get('uid', ''),
                            username=user.get('unique_id', ''),
                            nickname=user.get('nickname', ''),
                            scraped_at=datetime.now().isoformat()
                        ))
                        seen_ids.add(c_id)
                        self.stats["comments_scraped"] += 1
                except:
                    pass
            
            logger.info(f"✓ Scraped {len(comments)} comments for video {video_id}")
            return comments
            
        except Exception as e:
            logger.error(f"Error scraping comments for video {video_id}: {e}")
            return comments
            
        finally:
            try:
                if self._comment_handler:
                    self.page.remove_listener("response", self._comment_handler)
                    self._comment_handler = None
            except:
                pass

    async def get_video_metadata_from_network(self, url: str) -> Optional[VideoData]:
        """
        Navigate to video URL and capture metadata from API endpoints or hydration data.
        Tries network interception first, falls back to page data extraction.
        """
        self.intercepted_video_detail = None
        self.captured_initial_data = None

        try:
            path_parts = urlparse(url).path.strip('/').split('/')
            username = path_parts[0].replace('@', '') if len(path_parts) > 0 else "unknown"
            video_id = path_parts[2] if len(path_parts) > 2 else None
        except Exception:
            username = "unknown"
            video_id = None

        async def detail_listener(response: Response):
            # Broader set of potential API endpoints
            targets = [
                "api/item_detail",
                "api/iteminfo", 
                "api/general/search/single",
                "api/item/detail",
                "api/post/item_list",
                "api/recommend/item_list"
            ]
            
            if response.status == 200 and any(t in response.url for t in targets):
                try:
                    data = await response.json()
                    item = None
                    
                    if isinstance(data, dict):
                        # Try multiple paths to find item data
                        if data.get("itemInfo") and data["itemInfo"].get("itemStruct"):
                            item = data["itemInfo"]["itemStruct"]
                        elif data.get("itemStruct"):
                            item = data.get("itemStruct")
                        elif data.get("itemList") and len(data["itemList"]) > 0:
                            # Sometimes returned in a list
                            item = data["itemList"][0]
                        elif data.get("data") and isinstance(data["data"], dict):
                            if data["data"].get("item"):
                                item = data["data"]["item"]
                    
                    if item and isinstance(item, dict):
                        self.intercepted_video_detail = item
                        logger.debug(f"🎯 Captured video metadata from API: {response.url[:80]}")
                except Exception as e:
                    logger.debug(f"Error parsing response from {response.url[:60]}: {e}")

        self._detail_handler = detail_listener
        self.page.on("response", self._detail_handler)

        # Inject script BEFORE navigation to capture initial data as soon as it appears
        await self.page.add_init_script("""
            (() => {
                console.log('[CAPTURE] Init script loaded');
                window.__CAPTURED_INITIAL_DATA__ = null;
                let captured = false;
                
                // Poll for window.__UNIVERSAL_DATA_FOR_REHYDRATION__ to appear
                const pollInterval = setInterval(() => {
                    if (captured) {
                        clearInterval(pollInterval);
                        return;
                    }
                    
                    // Check if the object exists
                    const hasUNIVERSAL = typeof window.__UNIVERSAL_DATA_FOR_REHYDRATION__ !== 'undefined';
                    const hasSIGI = typeof window.SIGI_STATE !== 'undefined';
                    
                    if (hasUNIVERSAL || hasSIGI) {
                        console.log('[CAPTURE] Data object detected!');
                        const source = window.__UNIVERSAL_DATA_FOR_REHYDRATION__ || window.SIGI_STATE;
                        
                        try {
                            // Try to extract data immediately before it becomes a Proxy
                            const result = {};
                            
                            // Try to access __DEFAULT_SCOPE__ via bracket notation
                            const defaultScope = source['__DEFAULT_SCOPE__'];
                            if (defaultScope) {
                                console.log('[CAPTURE] Found __DEFAULT_SCOPE__');
                                result.__DEFAULT_SCOPE__ = {};
                                
                                const videoDetail = defaultScope['webapp.video-detail'];
                                if (videoDetail) {
                                    result.__DEFAULT_SCOPE__['webapp.video-detail'] = videoDetail;
                                    console.log('[CAPTURE] ✓ Captured webapp.video-detail');
                                }
                                
                                const itemDetail = defaultScope['webapp.item-detail'];
                                if (itemDetail) {
                                    result.__DEFAULT_SCOPE__['webapp.item-detail'] = itemDetail;
                                    console.log('[CAPTURE] ✓ Captured webapp.item-detail');
                                }
                            }
                            
                            // Try ItemModule
                            const itemModule = source['ItemModule'];
                            if (itemModule) {
                                result.ItemModule = itemModule;
                                console.log('[CAPTURE] ✓ Captured ItemModule');
                            }
                            
                            if (Object.keys(result).length > 0) {
                                window.__CAPTURED_INITIAL_DATA__ = result;
                                captured = true;
                                clearInterval(pollInterval);
                                console.log('[CAPTURE] ✓ Capture complete! Keys:', Object.keys(result));
                            } else {
                                console.log('[CAPTURE] Object found but no data extracted');
                            }
                        } catch(e) {
                            console.log('[CAPTURE] ✗ Error during capture:', e.message);
                        }
                    }
                }, 50); // Check every 50ms
                
                // Stop after 10 seconds
                setTimeout(() => {
                    if (!captured) {
                        console.log('[CAPTURE] Timeout - data not found');
                        clearInterval(pollInterval);
                    }
                }, 10000);
            })();
        """)

        try:
            logger.debug(f"Navigating to {url}")
            await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            
            # Wait a bit longer for API calls
            await asyncio.sleep(2)

            # Check if we got data from API
            for _ in range(8):
                if self.intercepted_video_detail:
                    break
                await asyncio.sleep(0.5)

            if self.intercepted_video_detail:
                return self._parse_video_data(self.intercepted_video_detail, username)
            
            # Fallback 1: Check captured initial data from page load
            logger.debug("API not captured, checking captured initial data...")
            captured_data = await self.page.evaluate("() => window.__CAPTURED_INITIAL_DATA__")
            
            if captured_data and isinstance(captured_data, dict):
                logger.debug(f"✓ Got captured initial data, keys: {list(captured_data.keys())}")
                item = None
                
                # Method 1: Check ItemModule with video_id
                if video_id:
                    item_module = captured_data.get('ItemModule', {})
                    if isinstance(item_module, dict) and video_id in item_module:
                        item = item_module[video_id]
                        logger.debug(f"✓ Found in captured ItemModule[{video_id}]")
                
                # Method 2: Check __DEFAULT_SCOPE__
                if not item:
                    default_scope = captured_data.get('__DEFAULT_SCOPE__', {})
                    if isinstance(default_scope, dict):
                        # Try video-detail
                        video_detail = default_scope.get('webapp.video-detail', {})
                        if video_detail.get('itemInfo', {}).get('itemStruct'):
                            item = video_detail['itemInfo']['itemStruct']
                            logger.debug("✓ Found in captured webapp.video-detail")
                        
                        # Try item-detail
                        if not item:
                            item_detail = default_scope.get('webapp.item-detail', {})
                            if item_detail.get('itemInfo', {}).get('itemStruct'):
                                item = item_detail['itemInfo']['itemStruct']
                                logger.debug("✓ Found in captured webapp.item-detail")
                
                # Method 3: Search all ItemModule values
                if not item:
                    item_module = captured_data.get('ItemModule', {})
                    if isinstance(item_module, dict):
                        for vid, data in item_module.items():
                            if isinstance(data, dict) and data.get('id') == video_id:
                                item = data
                                logger.debug(f"✓ Found by searching captured ItemModule")
                                break
                
                if item and isinstance(item, dict):
                    logger.debug("✓ Extracted from captured initial data")
                    return self._parse_video_data(item, username)
            
            # Fallback 2: Try extracting from page hydration data (old method)
            logger.debug("No captured data, trying page extraction...")
            page_data = await self._extract_page_data()
            
            if page_data:
                logger.debug(f"Page data keys: {list(page_data.keys())[:10]}")
                item = None
                
                # Method 1: Check ItemModule with video_id
                if video_id:
                    item_module = page_data.get('ItemModule', {})
                    if isinstance(item_module, dict) and video_id in item_module:
                        item = item_module[video_id]
                        logger.debug(f"✓ Found in ItemModule[{video_id}]")
                
                # Method 2: Check __DEFAULT_SCOPE__
                if not item:
                    default_scope = page_data.get('__DEFAULT_SCOPE__', {})
                    if isinstance(default_scope, dict):
                        logger.debug(f"Default scope keys: {list(default_scope.keys())[:5]}")
                        
                        # Try video-detail
                        video_detail = default_scope.get('webapp.video-detail', {})
                        if video_detail.get('itemInfo', {}).get('itemStruct'):
                            item = video_detail['itemInfo']['itemStruct']
                            logger.debug("✓ Found in webapp.video-detail")
                        
                        # Try item-detail
                        if not item:
                            item_detail = default_scope.get('webapp.item-detail', {})
                            if item_detail.get('itemInfo', {}).get('itemStruct'):
                                item = item_detail['itemInfo']['itemStruct']
                                logger.debug("✓ Found in webapp.item-detail")
                
                # Method 3: Search all ItemModule values
                if not item:
                    item_module = page_data.get('ItemModule', {})
                    if isinstance(item_module, dict):
                        for vid, data in item_module.items():
                            if isinstance(data, dict) and data.get('id') == video_id:
                                item = data
                                logger.debug(f"✓ Found by searching ItemModule")
                                break
                
                if item and isinstance(item, dict):
                    logger.debug("✓ Extracted from page hydration data")
                    return self._parse_video_data(item, username)
                else:
                    logger.debug("No valid item found in page data")
            else:
                logger.debug("Page data extraction returned None")
            
            logger.warning("⚠️ Could not capture metadata from API or page data")
            return None
            
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
            return None
        finally:
            if self._detail_handler:
                try:
                    self.page.remove_listener("response", self._detail_handler)
                except Exception:
                    pass
                self._detail_handler = None

    async def run_pipeline_2a_fetch_list(self, username: str, lookback_days: int):
        """Pipeline 2a: Save video URLs to txt using API interception only."""
        output_dir = Path("video_list") / datetime.now().strftime('%Y%m%d')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{username}.txt"

        milestone = datetime.now() - timedelta(days=lookback_days)
        logger.info(f"📍 Fetching list for @{username} (Milestone: {milestone.date()})")

        videos = await self.get_user_videos(username, 1000, milestone)

        if videos:
            with open(output_file, 'w', encoding='utf-8') as f:
                for v in videos:
                    f.write(f"{v.url}\n")
            logger.info(f"✅ Saved {len(videos)} URLs to {output_file}")
            return str(output_file)
        return None

    async def run_pipeline_2b_process_from_file(self, file_path: str):
        """Pipeline 2b: Process video URL list from file (network-only metadata)."""
        if not os.path.exists(file_path):
            logger.error("File not found.")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]

        logger.info(f"📂 Processing {len(urls)} videos from {file_path}")

        input_path = Path(file_path)
        username_from_file = input_path.stem
        date_str = datetime.now().strftime('%Y%m%d')
        output_base = Path(f"comments_data/{date_str}/{username_from_file}")
        output_base.mkdir(parents=True, exist_ok=True)

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

                save_path = output_base / f"{video_data.video_id}.json"
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

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
        max_comments_per_video: int = 500
    ):
        """
        PIPELINE 2: Deep scraping with comments and labels
        
        1. Fetch recent videos (last N days)
        2. Visit each video individually
        3. Capture diversification labels via network interception
        4. Capture comments
        5. Save to: comments_data/{date}/{username}/{video_id}.json
        """
        logger.info("=" * 70)
        logger.info(f"🚀 PIPELINE 2: Deep scrape for @{username} (Last {lookback_days} days)")
        logger.info("=" * 70)
        
        # Setup output directory
        date_str = datetime.now().strftime('%Y%m%d')
        base_dir = Path("comments_data") / date_str / username
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
                    "comments_count": len(comments),
                    "comments": [asdict(c) for c in comments]
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
    
    logger.info("=" * 70)
    logger.info("PIPELINE 2: DEEP SCRAPING WITH COMMENTS")
    logger.info("=" * 70)
    logger.info(f"Profiles: {len(usernames)}")
    logger.info(f"Lookback: {lookback_days} days")
    
    scraper = PlaywrightScraper(
        headless=False,
        restart_browser_every=5  # More aggressive for Pipeline 2
    )
    
    try:
        await scraper.start()
        
        for idx, username in enumerate(usernames):
            try:
                await scraper.run_pipeline_2_detailed(username, lookback_days)
                
                # Restart between users
                if username != usernames[-1]:
                    await scraper.restart_browser()
                    await asyncio.sleep(10)
                    
            except Exception as e:
                logger.error(f"Failed to process {username}: {e}")
                
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
    else:
        print("Invalid pipeline selection. Choose 1 or 2.")


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())