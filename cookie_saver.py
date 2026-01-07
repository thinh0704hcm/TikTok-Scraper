"""
TikTok Cookie Manager

Multiple methods to save and load TikTok cookies:
1. Manual browser export (EditThisCookie extension)
2. Interactive login with Playwright (recommended)
3. Chrome DevTools manual export
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CookieManager')

# ============================================================================
# METHOD 1: INTERACTIVE LOGIN WITH PLAYWRIGHT (RECOMMENDED)
# ============================================================================

async def save_cookies_interactive():
    """
    Opens a browser window for you to manually log in to TikTok.
    After you log in, press Enter to save the cookies.
    
    This is the most reliable method.
    """
    print("\n" + "="*70)
    print("TikTok Cookie Saver - Interactive Login")
    print("="*70)
    print("\nInstructions:")
    print("1. A browser window will open")
    print("2. Log in to TikTok manually")
    print("3. Browse around to ensure you're logged in")
    print("4. Come back to this terminal and press Enter")
    print("5. Cookies will be saved automatically")
    print("\nPress Enter to start...")
    input()
    
    async with async_playwright() as p:
        # Launch browser with normal user settings
        browser = await p.chromium.launch(
            headless=False,  # MUST be visible for manual login
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
            ]
        )
        
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        )
        
        page = await context.new_page()
        
        # Go to TikTok
        print("\n🌐 Opening TikTok...")
        await page.goto('https://www.tiktok.com/')
        await asyncio.sleep(2)
        
        print("\n✋ Please log in to TikTok in the browser window")
        print("   Once logged in, press Enter here to save cookies...")
        input()
        
        # Get all cookies from the browser
        cookies = await context.cookies()
        
        if not cookies:
            print("\n❌ No cookies found! Did you log in?")
            await browser.close()
            return False
        
        # Check if we have the important TikTok cookies
        cookie_names = {c['name'] for c in cookies}
        important_cookies = ['sessionid', 'sid_tt', 'sid_guard']
        has_session = any(c in cookie_names for c in important_cookies)
        
        if not has_session:
            print(f"\n⚠️ Warning: No session cookies found!")
            print(f"   Found cookies: {cookie_names}")
            print(f"   You may not be logged in properly.")
        
        # Save cookies to file
        cookie_file = Path('tiktok_cookies.json')
        with open(cookie_file, 'w') as f:
            json.dump(cookies, f, indent=2)
        
        print(f"\n✅ Saved {len(cookies)} cookies to {cookie_file}")
        print(f"   Session cookies present: {has_session}")
        
        # Also create a backup with timestamp
        backup_file = Path(f'tiktok_cookies_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(backup_file, 'w') as f:
            json.dump(cookies, f, indent=2)
        print(f"   Backup saved to {backup_file}")
        
        await browser.close()
        
        print("\n" + "="*70)
        print("Cookie Information:")
        print("="*70)
        print(f"Total cookies: {len(cookies)}")
        print(f"Session cookies: {[c for c in important_cookies if c in cookie_names]}")
        print(f"Domains: {set(c['domain'] for c in cookies)}")
        
        return True


# ============================================================================
# METHOD 2: LOAD AND VALIDATE EXISTING COOKIES
# ============================================================================

def load_and_validate_cookies(cookie_file: str = 'tiktok_cookies.json') -> bool:
    """
    Load cookies from file and validate they're in the correct format.
    Returns True if cookies are valid.
    """
    cookie_path = Path(cookie_file)
    
    if not cookie_path.exists():
        print(f"❌ Cookie file not found: {cookie_file}")
        return False
    
    try:
        with open(cookie_path, 'r') as f:
            cookies = json.load(f)
        
        if not isinstance(cookies, list):
            print(f"❌ Invalid format: cookies should be a list, got {type(cookies)}")
            return False
        
        # Check required fields
        required_fields = ['name', 'value', 'domain']
        for i, cookie in enumerate(cookies):
            missing = [f for f in required_fields if f not in cookie]
            if missing:
                print(f"❌ Cookie {i} missing required fields: {missing}")
                return False
        
        # Check for session cookies
        cookie_names = {c['name'] for c in cookies}
        important_cookies = ['sessionid', 'sid_tt', 'sid_guard', 'msToken']
        session_cookies = [c for c in important_cookies if c in cookie_names]
        
        print(f"\n✅ Cookie file is valid!")
        print(f"   Total cookies: {len(cookies)}")
        print(f"   Session cookies: {session_cookies}")
        print(f"   Domains: {set(c['domain'] for c in cookies)}")
        
        # Check expiry
        now = datetime.now().timestamp()
        expired = [c for c in cookies if 'expires' in c and c['expires'] != -1 and c['expires'] < now]
        if expired:
            print(f"   ⚠️ Warning: {len(expired)} cookies have expired")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


# ============================================================================
# METHOD 3: CONVERT FROM EDITTHISCOOKIE FORMAT
# ============================================================================

def convert_editthiscookie_to_playwright(input_file: str, output_file: str = 'tiktok_cookies.json'):
    """
    Convert cookies from EditThisCookie browser extension format to Playwright format.
    
    To use EditThisCookie:
    1. Install EditThisCookie extension: https://chrome.google.com/webstore
    2. Go to tiktok.com (logged in)
    3. Click EditThisCookie icon
    4. Click "Export" (bottom right)
    5. Save to a file
    6. Run this function
    """
    try:
        with open(input_file, 'r') as f:
            content = f.read()
        
        # EditThisCookie exports as JSON array
        cookies = json.loads(content)
        
        # Convert to Playwright format
        playwright_cookies = []
        for cookie in cookies:
            pw_cookie = {
                'name': cookie.get('name', ''),
                'value': cookie.get('value', ''),
                'domain': cookie.get('domain', ''),
                'path': cookie.get('path', '/'),
                'expires': cookie.get('expirationDate', -1),
                'httpOnly': cookie.get('httpOnly', False),
                'secure': cookie.get('secure', False),
            }
            
            # Add sameSite if present
            if 'sameSite' in cookie:
                sameSite = cookie['sameSite']
                if sameSite in ['Strict', 'Lax', 'None']:
                    pw_cookie['sameSite'] = sameSite
            
            playwright_cookies.append(pw_cookie)
        
        # Save converted cookies
        with open(output_file, 'w') as f:
            json.dump(playwright_cookies, f, indent=2)
        
        print(f"✅ Converted {len(playwright_cookies)} cookies")
        print(f"   Saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False


# ============================================================================
# METHOD 4: MANUALLY CREATE FROM DEVTOOLS
# ============================================================================

def create_cookies_from_devtools_export():
    """
    Instructions for manually exporting cookies from Chrome DevTools.
    
    This is useful if you can't use extensions or Playwright.
    """
    instructions = """
╔══════════════════════════════════════════════════════════════════════╗
║           Manual Cookie Export from Chrome DevTools                  ║
╚══════════════════════════════════════════════════════════════════════╝

1. Open Chrome/Brave and go to TikTok (logged in)

2. Press F12 to open DevTools

3. Go to "Application" tab (or "Storage" in Firefox)

4. Expand "Cookies" in the left sidebar

5. Click on "https://www.tiktok.com"

6. Look for these important cookies and copy their values:
   - sessionid
   - sid_tt
   - sid_guard
   - msToken
   - tt_chain_token

7. Open the Console tab in DevTools

8. Paste this code and press Enter:

   copy(JSON.stringify(
     document.cookie.split('; ').map(c => {
       const [name, value] = c.split('=');
       return {
         name: name,
         value: value,
         domain: '.tiktok.com',
         path: '/',
         expires: -1,
         httpOnly: false,
         secure: true,
         sameSite: 'None'
       };
     }),
     null,
     2
   ));

9. The cookies are now copied to your clipboard

10. Create a file called 'tiktok_cookies.json' and paste the content

11. Run: python cookie_saver.py --validate

════════════════════════════════════════════════════════════════════════
"""
    print(instructions)


# ============================================================================
# METHOD 5: TEST COOKIES WITH PLAYWRIGHT
# ============================================================================

async def test_cookies(cookie_file: str = 'tiktok_cookies.json'):
    """
    Test if cookies work by loading TikTok with them.
    """
    print("\n" + "="*70)
    print("Testing TikTok Cookies")
    print("="*70)
    
    cookie_path = Path(cookie_file)
    if not cookie_path.exists():
        print(f"❌ Cookie file not found: {cookie_file}")
        return False
    
    try:
        with open(cookie_path, 'r') as f:
            cookies = json.load(f)
        
        print(f"📦 Loaded {len(cookies)} cookies")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            
            # Add cookies
            await context.add_cookies(cookies)
            print("✓ Cookies added to browser")
            
            page = await context.new_page()
            
            # Navigate to TikTok
            print("\n🌐 Loading TikTok...")
            await page.goto('https://www.tiktok.com/')
            await asyncio.sleep(3)
            
            # Check if logged in by looking for profile button or username
            try:
                # Try to find user profile indicators
                profile_selectors = [
                    '[data-e2e="profile-icon"]',
                    '[data-e2e="nav-profile"]',
                    'a[href*="/profile/"]',
                ]
                
                logged_in = False
                for selector in profile_selectors:
                    try:
                        await page.wait_for_selector(selector, timeout=5000)
                        logged_in = True
                        break
                    except:
                        continue
                
                if logged_in:
                    print("✅ SUCCESS: Cookies work! You are logged in.")
                else:
                    print("⚠️ UNCERTAIN: Could not confirm login status")
                    print("   The browser window is open - check manually")
                
                print("\nBrowser will stay open for 10 seconds for manual verification...")
                await asyncio.sleep(10)
                
            except Exception as e:
                print(f"❌ Error checking login status: {e}")
                return False
            
            await browser.close()
            return logged_in
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


# ============================================================================
# MAIN CLI
# ============================================================================

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='TikTok Cookie Manager')
    parser.add_argument('--save', action='store_true', help='Save cookies interactively (recommended)')
    parser.add_argument('--test', action='store_true', help='Test existing cookies')
    parser.add_argument('--validate', action='store_true', help='Validate cookie file format')
    parser.add_argument('--convert', type=str, help='Convert EditThisCookie export to Playwright format')
    parser.add_argument('--instructions', action='store_true', help='Show manual export instructions')
    parser.add_argument('--file', type=str, default='tiktok_cookies.json', help='Cookie file path')
    
    args = parser.parse_args()
    
    if args.save:
        await save_cookies_interactive()
    elif args.test:
        await test_cookies(args.file)
    elif args.validate:
        load_and_validate_cookies(args.file)
    elif args.convert:
        convert_editthiscookie_to_playwright(args.convert, args.file)
    elif args.instructions:
        create_cookies_from_devtools_export()
    else:
        print("TikTok Cookie Manager\n")
        print("Usage:")
        print("  python cookie_saver.py --save          # Save cookies interactively (recommended)")
        print("  python cookie_saver.py --test          # Test existing cookies")
        print("  python cookie_saver.py --validate      # Validate cookie format")
        print("  python cookie_saver.py --convert FILE  # Convert EditThisCookie export")
        print("  python cookie_saver.py --instructions  # Show manual export guide")
        print("\nFor help: python cookie_saver.py --help")


if __name__ == '__main__':
    asyncio.run(main())