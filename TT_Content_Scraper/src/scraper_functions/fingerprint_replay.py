# fingerprint_replay.py
import asyncio
from patchright.async_api import async_playwright

# PASTE YOUR CAPTURED FINGERPRINT HERE
MY_FINGERPRINT = {
    "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    "platform": "Win32",
    "hardwareConcurrency": 14,
    "deviceMemory": 2,
    "maxTouchPoints": 10,
    "languages": [
        "en-US"
    ],
    "webgl": {
        "vendor": "Google Inc. (AMD)",
        "renderer": "ANGLE (AMD, AMD Radeon(TM) Graphics (0x00001681) Direct3D11 vs_5_0 ps_5_0, D3D11)",
        "version": "WebGL 1.0 (OpenGL ES 2.0 Chromium)",
        "shadingLanguageVersion": "WebGL GLSL ES 1.0 (OpenGL ES GLSL ES 1.0 Chromium)"
    },
    "screen": {
        "width": 1680,
        "height": 1050,
        "availWidth": 1680,
        "availHeight": 1050,
        "colorDepth": 24,
        "pixelDepth": 24
    },
    "plugins": [
        {
        "name": "Chromium PDF Viewer",
        "description": "Portable Document Format",
        "filename": "internal-pdf-viewer",
        "length": 2
        },
        {
        "name": "Microsoft Edge PDF Viewer",
        "description": "Portable Document Format",
        "filename": "internal-pdf-viewer",
        "length": 2
        },
        {
        "name": "PDF Viewer",
        "description": "Portable Document Format",
        "filename": "internal-pdf-viewer",
        "length": 2
        },
        {
        "name": "GiwYUpzh",
        "description": "OmyCJEKFpc1asWq0iRnb057d1asWq0iZ",
        "filename": "QnTRQnTRQnTJr8Hi",
        "length": 2
        },
        {
        "name": "Online PDF and PS Viewer",
        "description": "Portable Document Format",
        "filename": "YrdOmy4cWTwBAIjZMOHLkaVKs15FCgQI",
        "length": 2
        },
        {
        "name": "dOuf2bN",
        "description": "yCBfXyhQQv2EpUKkaVKs9e2bVSJr0iZ",
        "filename": "JIMlxBAIjZUxg3E",
        "length": 2
        },
        {
        "name": "WebKit built-in PDF",
        "description": "Portable Document Format",
        "filename": "internal-pdf-viewer",
        "length": 2
        }
    ],
    "timezone": "Asia/Saigon",
    "timezoneOffset": -420
}

async def create_stealth_context():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=['--no-sandbox']
        )
        
        context = await browser.new_context(
            viewport={'width': MY_FINGERPRINT['screen']['width'], 
                     'height': MY_FINGERPRINT['screen']['height']},
            screen={'width': MY_FINGERPRINT['screen']['width'],
                   'height': MY_FINGERPRINT['screen']['height']},
            user_agent=MY_FINGERPRINT['userAgent'],
            locale='en-US',
            timezone_id=MY_FINGERPRINT['timezone'],
        )
        
        # Inject your real fingerprint
        await context.add_init_script(f"""
            (() => {{
                const FINGERPRINT = {MY_FINGERPRINT};
                
                // WebGL
                const origGetParam = WebGLRenderingContext.prototype.getParameter;
                WebGLRenderingContext.prototype.getParameter = function(p) {{
                    if (p === 37445 || p === 7936) return FINGERPRINT.webgl.vendor;
                    if (p === 37446 || p === 7937) return FINGERPRINT.webgl.renderer;
                    return origGetParam.call(this, p);
                }};
                
                // Platform
                Object.defineProperty(navigator, 'platform', {{
                    get: () => FINGERPRINT.platform
                }});
                
                // Hardware
                Object.defineProperty(navigator, 'hardwareConcurrency', {{
                    get: () => FINGERPRINT.hardwareConcurrency
                }});
                
                // Remove webdriver
                Object.defineProperty(navigator, 'webdriver', {{
                    get: () => undefined
                }});
                
                console.log('✓ Real fingerprint loaded');
            }})();
        """)
        
        return context, browser