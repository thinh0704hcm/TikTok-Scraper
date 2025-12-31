"""
Script để lấy thông tin video TikTok từ link
Usage: python get_video_info.py <video_url>
Example: python get_video_info.py https://www.tiktok.com/@username/video/123456789
"""

import asyncio
import json
import re
import sys
from datetime import datetime
from playwright.async_api import async_playwright
from typing import Optional, Dict, Any


async def get_video_info(video_url: str) -> Optional[Dict[str, Any]]:
    """
    Lấy thông tin video TikTok từ URL.
    
    Args:
        video_url: URL của video TikTok (ví dụ: https://www.tiktok.com/@username/video/123456789)
    
    Returns:
        Dict chứa thông tin video hoặc None nếu lỗi
    """
    print(f"\n🔍 Đang lấy thông tin video...")
    print(f"   URL: {video_url}")
    
    # Parse video ID và username từ URL
    pattern = r'@([^/]+)/video/(\d+)'
    match = re.search(pattern, video_url)
    
    if not match:
        print("❌ URL không hợp lệ! Định dạng đúng: https://www.tiktok.com/@username/video/123456789")
        return None
    
    username = match.group(1)
    video_id = match.group(2)
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        
        page = await context.new_page()
        
        # Intercept API responses
        video_data = None
        
        async def handle_response(response):
            nonlocal video_data
            
            # Catch video detail API
            if '/api/item/detail/' in response.url or '/aweme/v1/feed/' in response.url:
                try:
                    data = await response.json()
                    
                    # Extract video info from response
                    if 'itemInfo' in data and 'itemStruct' in data['itemInfo']:
                        item = data['itemInfo']['itemStruct']
                    elif 'aweme_list' in data and len(data['aweme_list']) > 0:
                        item = data['aweme_list'][0]
                    else:
                        return
                    
                    # Extract stats
                    stats = item.get('stats', {})
                    author = item.get('author', {})
                    
                    video_data = {
                        'video_id': item.get('id') or video_id,
                        'author_username': author.get('uniqueId', username),
                        'author_nickname': author.get('nickname', ''),
                        'author_id': author.get('id', ''),
                        'description': item.get('desc', ''),
                        'create_time': datetime.fromtimestamp(item.get('createTime', 0)).isoformat(),
                        'create_timestamp': item.get('createTime', 0),
                        'views': stats.get('playCount', 0),
                        'likes': stats.get('diggCount', 0),
                        'comments': stats.get('commentCount', 0),
                        'shares': stats.get('shareCount', 0),
                        'collects': stats.get('collectCount', 0),
                        'music_title': item.get('music', {}).get('title', ''),
                        'hashtags': [tag.get('title', '') for tag in item.get('textExtra', []) if tag.get('hashtagName')],
                        'scraped_at': datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    print(f"   ⚠️  Error parsing response: {e}")
        
        page.on('response', handle_response)
        
        try:
            # Navigate to video page
            print("   🌐 Đang mở trang...")
            await page.goto(video_url, wait_until='networkidle', timeout=60000)
            
            # Wait longer for content to load
            print("   ⏳ Đang chờ video tải...")
            await page.wait_for_timeout(5000)
            
            # Try to get data from page if API didn't work
            if video_data is None:
                print("   📄 Đang đọc dữ liệu từ trang...")
                
                try:
                    # Wait for any video content to appear
                    await page.wait_for_selector('video', timeout=10000)
                    
                    # Try multiple selectors for stats
                    async def get_stat(selectors):
                        for selector in selectors:
                            try:
                                elem = page.locator(selector).first
                                if await elem.count() > 0:
                                    text = await elem.text_content(timeout=5000)
                                    return text.strip() if text else '0'
                            except:
                                continue
                        return '0'
                    
                    # Get stats with multiple selector options
                    views = await get_stat([
                        '[data-e2e="video-views"]',
                        'strong[data-e2e="video-views"]',
                        '[data-e2e="browse-video-views"]',
                        '[class*="view-count"]',
                        'strong[title*="view"]',
                        '[class*="ViewCount"]',
                        'span[class*="view"]',
                        'div[class*="VideoDescription"] strong:first-child'
                    ])
                    
                    # If views still 0, try to find it in page content
                    if views == '0':
                        try:
                            page_content = await page.content()
                            # Look for view count patterns in HTML
                            view_patterns = [
                                r'"playCount":(\d+)',
                                r'"viewCount":(\d+)',
                                r'playCount&quot;:(\d+)',
                                r'data-e2e="video-views"[^>]*>([^<]+)',
                            ]
                            for pattern in view_patterns:
                                match = re.search(pattern, page_content)
                                if match:
                                    views = match.group(1)
                                    print(f"   🔍 Tìm thấy views trong HTML: {views}")
                                    break
                        except:
                            pass
                    
                    likes = await get_stat([
                        '[data-e2e="like-count"]',
                        'strong[data-e2e="like-count"]',
                        '[data-e2e="browse-like-count"]'
                    ])
                    
                    comments = await get_stat([
                        '[data-e2e="comment-count"]',
                        'strong[data-e2e="comment-count"]',
                        '[data-e2e="browse-comment-count"]'
                    ])
                    
                    shares = await get_stat([
                        '[data-e2e="share-count"]',
                        'strong[data-e2e="share-count"]',
                        '[data-e2e="browse-share-count"]'
                    ])
                    
                    collects = await get_stat([
                        '[data-e2e="undefined-count"]',
                        'strong[data-e2e="undefined-count"]',
                        '[data-e2e="collect-count"]',
                        '[data-e2e="browse-collect-count"]',
                        'button[data-e2e="browse-collect"] strong'
                    ])
                    
                    # If collects still 0, try to find in HTML
                    if collects == '0':
                        try:
                            page_content = await page.content()
                            collect_patterns = [
                                r'"collectCount":(\d+)',
                                r'collectCount&quot;:(\d+)',
                            ]
                            for pattern in collect_patterns:
                                match = re.search(pattern, page_content)
                                if match:
                                    collects = match.group(1)
                                    print(f"   🔍 Tìm thấy collects trong HTML: {collects}")
                                    break
                        except:
                            pass
                    
                    # Get description
                    try:
                        desc_selectors = [
                            '[data-e2e="video-desc"]',
                            '[data-e2e="browse-video-desc"]',
                            'h1[data-e2e="browse-video-desc"]'
                        ]
                        desc = ''
                        for sel in desc_selectors:
                            try:
                                elem = page.locator(sel).first
                                if await elem.count() > 0:
                                    desc = await elem.text_content(timeout=3000)
                                    break
                            except:
                                continue
                    except:
                        desc = ''
                    
                    print(f"   📊 Đọc được: views={views}, likes={likes}, comments={comments}, shares={shares}")
                    
                    # If views is still 0 but we have other stats, try screenshot for debugging
                    if views == '0' and (likes != '0' or comments != '0'):
                        print("   ⚠️  Views = 0 nhưng có stats khác, đang debug...")
                        try:
                            # Try to find the views element by taking screenshot
                            await page.screenshot(path='debug_video_page.png', full_page=False)
                            print("   📸 Đã chụp màn hình debug_video_page.png")
                            
                            # Try to get all strong tags and find views
                            all_strongs = await page.locator('strong').all()
                            print(f"   🔍 Tìm thấy {len(all_strongs)} strong tags, đang kiểm tra...")
                            for strong in all_strongs[:20]:  # Check first 20
                                try:
                                    text = await strong.text_content(timeout=1000)
                                    if text and ('K' in text or 'M' in text or text.isdigit()):
                                        # Get parent context
                                        parent_text = await strong.evaluate('el => el.parentElement?.textContent || ""')
                                        if 'view' in parent_text.lower() or len(text) > 3:
                                            print(f"      • {text} (context: {parent_text[:50]})")
                                            # If this looks like a big number, might be views
                                            if views == '0' and ('M' in text or ('K' in text and float(text.replace('K','')) > 50)):
                                                views = text
                                                print(f"   ✅ Chọn {text} làm views (số lớn nhất)")
                                except:
                                    continue
                        except Exception as e:
                            print(f"   Debug error: {e}")
                    
                    # Parse numbers (handle K, M notation)
                    def parse_count(text):
                        if not text:
                            return 0
                        text = text.strip().upper()
                        if 'K' in text:
                            return int(float(text.replace('K', '')) * 1000)
                        elif 'M' in text:
                            return int(float(text.replace('M', '')) * 1000000)
                        else:
                            return int(text.replace(',', ''))
                    
                    video_data = {
                        'video_id': video_id,
                        'author_username': username,
                        'author_nickname': '',
                        'author_id': '',
                        'description': desc,
                        'create_time': '',
                        'create_timestamp': 0,
                        'views': parse_count(views),
                        'likes': parse_count(likes),
                        'comments': parse_count(comments),
                        'shares': parse_count(shares),
                        'collects': parse_count(collects),
                        'music_title': '',
                        'hashtags': [],
                        'scraped_at': datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    print(f"   ⚠️  Không thể đọc dữ liệu từ trang: {e}")
            
        except Exception as e:
            print(f"❌ Lỗi khi truy cập trang: {e}")
        
        finally:
            await browser.close()
    
    return video_data


def format_number(num):
    """Format number with commas"""
    return f"{num:,}"


async def main():
    """Main function"""
    print("=" * 70)
    print("🎬 TIKTOK VIDEO INFO SCRAPER")
    print("=" * 70)
    
    # Get URL from command line or user input
    if len(sys.argv) > 1:
        video_url = sys.argv[1].strip()
        print(f"\n📎 URL từ tham số: {video_url}")
    else:
        video_url = input("\n📎 Nhập link video TikTok: ").strip()
    
    if not video_url:
        print("❌ URL không được để trống!")
        print("\nUsage: python get_video_info.py <video_url>")
        print("Example: python get_video_info.py https://www.tiktok.com/@username/video/123456789")
        return
    
    # Get video info
    video_data = await get_video_info(video_url)
    
    if video_data:
        print("\n" + "=" * 70)
        print("✅ THÔNG TIN VIDEO")
        print("=" * 70)
        
        print(f"\n📹 Video ID: {video_data['video_id']}")
        print(f"👤 Author: @{video_data['author_username']}")
        if video_data['author_nickname']:
            print(f"   Nickname: {video_data['author_nickname']}")
        
        print(f"\n📝 Description:")
        print(f"   {video_data['description'][:200]}{'...' if len(video_data['description']) > 200 else ''}")
        
        if video_data['hashtags']:
            print(f"\n🏷️  Hashtags: {', '.join(['#' + tag for tag in video_data['hashtags']])}")
        
        if video_data['music_title']:
            print(f"🎵 Music: {video_data['music_title']}")
        
        print(f"\n📊 ENGAGEMENT METRICS:")
        print(f"   👁️  Views:    {format_number(video_data['views'])}")
        print(f"   ❤️  Likes:    {format_number(video_data['likes'])}")
        print(f"   💬 Comments: {format_number(video_data['comments'])}")
        print(f"   🔄 Shares:   {format_number(video_data['shares'])}")
        print(f"   ⭐ Collects: {format_number(video_data['collects'])}")
        
        if video_data['views'] > 0:
            engagement_rate = (video_data['likes'] + video_data['comments'] + video_data['shares']) / video_data['views'] * 100
            print(f"\n📈 Engagement Rate: {engagement_rate:.2f}%")
        
        if video_data['create_time']:
            print(f"\n⏰ Posted: {video_data['create_time']}")
        
        print(f"🕐 Scraped: {video_data['scraped_at']}")
        
        print("\n💾 Lưu dữ liệu vào file JSON")
        from pathlib import Path
        
        # Create directory structure: specific_video_data/video_{id}/
        video_dir = Path('specific_video_data') / f"video_{video_data['video_id']}"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with datetime filename
        filename = video_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(video_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu vào: {filename}")
    
    else:
        print("\n❌ Không thể lấy thông tin video!")
        print("Có thể do:")
        print("  - Video không tồn tại hoặc đã bị xóa")
        print("  - TikTok yêu cầu đăng nhập")
        print("  - Vấn đề kết nối mạng")


if __name__ == "__main__":
    asyncio.run(main())
