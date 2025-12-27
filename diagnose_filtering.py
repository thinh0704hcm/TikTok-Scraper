import json
from pathlib import Path
from datetime import datetime
import pandas as pd

print("Loading raw data...")
all_records = []

for json_file in Path('video_data/list32/90').rglob('*videos_raw_*.json'):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            for video in data:
                try:
                    stats = video.get('stats', {})
                    create_time = video.get('create_time')
                    if isinstance(create_time, str):
                        posted_at = pd.to_datetime(create_time)
                    else:
                        create_timestamp = video.get('create_timestamp')
                        posted_at = pd.to_datetime(create_timestamp, unit='s')
                    
                    scraped_at = pd.to_datetime(video.get('scraped_at'))
                    
                    all_records.append({
                        'video_id': str(video.get('video_id')),
                        'posted_at': posted_at,
                        'scraped_at': scraped_at,
                        't_since_post': (scraped_at - posted_at).total_seconds() / 3600,
                    })
                except:
                    pass

df = pd.DataFrame(all_records)
print(f"Total records loaded: {len(df)}")
print(f"Unique videos: {df['video_id'].nunique()}")

# Check filtering criteria
print("\n=== Filtering Analysis ===")

video_stats = df.groupby('video_id').agg({
    't_since_post': ['min', 'max', 'count']
}).reset_index()
video_stats.columns = ['video_id', 'min_hours', 'max_hours', 'count']

print(f"\nTotal videos: {len(video_stats)}")

# Filter 1: tracked from ≤8h
early_videos = video_stats[video_stats['min_hours'] <= 8]
print(f"Videos tracked from ≤8h: {len(early_videos)}")

# Filter 2: tracked to ≥24h
long_videos = video_stats[video_stats['max_hours'] >= 24]
print(f"Videos tracked to ≥24h: {len(long_videos)}")

# Combined
suitable_videos = video_stats[(video_stats['min_hours'] <= 8) & (video_stats['max_hours'] >= 24)]
print(f"Videos with BOTH (≤8h start AND ≥24h end): {len(suitable_videos)}")

# Check early measurements
print("\n=== Early Measurement Analysis (for suitable videos) ===")
if len(suitable_videos) > 0:
    for vid in suitable_videos['video_id'].head(10):
        video_data = df[df['video_id'] == vid].sort_values('scraped_at')
        early_data = video_data[video_data['t_since_post'] <= 8]
        print(f"Video {vid}: {len(early_data)} measurements in 0-8h range (need ≥8 for sequences)")
else:
    print("No suitable videos found!")

print("\n=== Sample video timeline ===")
sample_vid = video_stats.iloc[0]['video_id']
sample_data = df[df['video_id'] == sample_vid].sort_values('scraped_at')
print(f"\nVideo: {sample_vid}")
print(f"Measurements: {len(sample_data)}")
print(f"Time range: {sample_data['t_since_post'].min():.1f}h to {sample_data['t_since_post'].max():.1f}h")
print(f"\nFirst 10 measurements:")
for idx, row in sample_data.head(10).iterrows():
    print(f"  {row['t_since_post']:.2f}h - scraped at {row['scraped_at']}")
