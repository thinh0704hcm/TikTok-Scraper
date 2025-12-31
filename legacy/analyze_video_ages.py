import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

video_info = defaultdict(lambda: {'min_hours': float('inf'), 'max_hours': 0, 'snapshots': 0})

for json_file in Path('video_data/list32/90').rglob('*videos_raw_*.json'):
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
        videos = data if isinstance(data, list) else [data]
        
        for v in videos:
            vid = v['video_id']
            posted = datetime.fromisoformat(v['create_time'].replace('Z', '+00:00'))
            scraped = datetime.fromisoformat(v['scraped_at'].replace('Z', '+00:00'))
            hours_since_post = (scraped - posted).total_seconds() / 3600
            
            video_info[vid]['min_hours'] = min(video_info[vid]['min_hours'], hours_since_post)
            video_info[vid]['max_hours'] = max(video_info[vid]['max_hours'], hours_since_post)
            video_info[vid]['snapshots'] += 1

# Analyze
videos_early = []  # Started tracking within first 24h after posting
videos_mature = []  # All measurements are >24h after posting

for vid, info in video_info.items():
    span = info['max_hours'] - info['min_hours']
    
    if info['min_hours'] <= 24:
        videos_early.append((vid, info))
    if info['min_hours'] >= 24:
        videos_mature.append((vid, info))

print(f"Total videos tracked: {len(video_info)}")
print(f"\nVideos tracked from early stage (≤24h after post): {len(videos_early)}")
print(f"Videos tracked only when mature (>24h after post): {len(videos_mature)}")

print(f"\n=== Tracking Duration Stats ===")
spans = [(info['max_hours'] - info['min_hours']) for info in video_info.values()]
print(f"Min tracking duration: {min(spans):.1f}h")
print(f"Max tracking duration: {max(spans):.1f}h")
print(f"Median tracking duration: {sorted(spans)[len(spans)//2]:.1f}h")

print(f"\n=== Early-stage videos (tracked from ≤24h) ===")
if videos_early:
    for vid, info in sorted(videos_early, key=lambda x: x[1]['min_hours'])[:10]:
        span = info['max_hours'] - info['min_hours']
        print(f"Video {vid}: {info['min_hours']:.1f}h → {info['max_hours']:.1f}h (span: {span:.1f}h, {info['snapshots']} snapshots)")
else:
    print("None found - all videos were already old when scraping started")
