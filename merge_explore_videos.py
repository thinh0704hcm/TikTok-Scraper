import json
import os
import glob
from pathlib import Path
from datetime import datetime

def merge_explore_videos():
    """
    Merge all videos from _runXX.json files in video_metadata/explore
    into all.json in video_metadata/random_explore_7_days/20260111
    """
    
    # Define paths
    source_folder = Path("video_metadata/explore")
    target_folder = Path("video_metadata/random_explore_7_days/20260111")
    target_file = target_folder / "all.json"
    
    # Create target folder if it doesn't exist
    target_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all _runXX.json files
    pattern = str(source_folder / "*_run[0-9][0-9].json")
    run_files = sorted(glob.glob(pattern))
    
    print(f"Found {len(run_files)} run files to process")
    
    # Track unique videos using video_id
    unique_videos = {}
    duplicate_count = 0
    
    # Process each run file
    for run_file in run_files:
        print(f"Processing: {Path(run_file).name}")
        
        try:
            with open(run_file, 'r', encoding='utf-8') as f:
                videos = json.load(f)
            
            # Track videos from this file
            file_added = 0
            file_duplicates = 0
            
            for video in videos:
                video_id = video.get('video_id')
                
                if video_id:
                    if video_id not in unique_videos:
                        unique_videos[video_id] = video
                        file_added += 1
                    else:
                        duplicate_count += 1
                        file_duplicates += 1
            
            print(f"  - Added: {file_added}, Duplicates: {file_duplicates}")
            
        except Exception as e:
            print(f"  - Error processing {run_file}: {e}")
    
    # Convert to list for JSON output
    all_videos = list(unique_videos.values())
    
    # Sort by create_timestamp if available
    all_videos.sort(key=lambda x: x.get('create_timestamp', 0), reverse=True)
    
    # Write to target file
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(all_videos, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total files processed: {len(run_files)}")
    print(f"Total unique videos: {len(unique_videos)}")
    print(f"Total duplicates found: {duplicate_count}")
    print(f"Output file: {target_file}")
    print(f"File size: {os.path.getsize(target_file) / (1024*1024):.2f} MB")
    print("="*60)
    
    return len(unique_videos)

if __name__ == "__main__":
    unique_count = merge_explore_videos()
    print(f"\n✓ Successfully merged {unique_count} unique videos!")
