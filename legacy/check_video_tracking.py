import json
from pathlib import Path
from datetime import datetime

# Find first video
target_video = None
for json_file in Path('video_data/list32/90').rglob('*videos_raw_*.json'):
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
        videos = data if isinstance(data, list) else [data]
        if videos:
            target_video = videos[0]['video_id']
            break

print(f'Tracking video: {target_video}\n')

# Track this video across all scraping sessions
snapshots = []
for json_file in Path('video_data/list32/90').rglob('*videos_raw_*.json'):
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
        videos = data if isinstance(data, list) else [data]
        for v in videos:
            if v['video_id'] == target_video:
                snapshots.append({
                    'posted': v['create_time'],
                    'scraped': v['scraped_at'],
                    'views': v['stats']['playCount']
                })

print(f'Found {len(snapshots)} snapshots')
print('\nFirst 10 snapshots:')
for i, s in enumerate(snapshots[:10]):
    posted = datetime.fromisoformat(s['posted'].replace('Z', '+00:00'))
    scraped = datetime.fromisoformat(s['scraped'].replace('Z', '+00:00'))
    hours_since_post = (scraped - posted).total_seconds() / 3600
    print(f"{i+1}. Scraped: {s['scraped'][:19]} | Hours since post: {hours_since_post:.1f}h | Views: {s['views']:,}")
