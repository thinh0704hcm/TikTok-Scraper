"""
LSTM Data Preparation Pipeline for TikTok Video Analytics (RELAXED VERSION)

This script processes raw TikTok video data with forgiving criteria to maximize usable data.
Follows the relaxed requirements in process_data_for_lstm.md
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import pickle
from typing import List, Dict, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class LSTMDataPreparator:
    """
    Prepares TikTok video data for LSTM prediction model.
    
    Model Requirements (from instructor):
    - Input: Sequence of 3 snapshots (hour 1, 2, 3 after posting) + time_horizon
    - Each snapshot has 4 metrics: views, likes, shares, comments
    - time_horizon: days ahead to predict (1-7 days = 24-168 hours)
    - Output: 4 metrics at specified time_horizon
    
    Data Processing:
    - Hourly resampling (0-168 hours = 7 days)
    - Linear interpolation for missing hours
    - Data augmentation (synthetic videos) due to limited real data
    - Supervised learning format
    """
    
    def __init__(self, 
                 video_data_dir: str,
                 output_dir: str = "data_processing/processed",
                 max_days: int = 7):
        """
        Initialize the data preparator.
        
        Args:
            video_data_dir: Root directory containing video data
            output_dir: Directory to save processed data and artifacts
            max_days: Maximum days since posting to include (7 days = 168h)
        """
        self.video_data_dir = Path(video_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # FIXED CONFIGURATION (theo yêu cầu thầy)
        self.sequence_length = 3  # 3 snapshots: hour 1, 2, 3
        self.prediction_horizons = [24, 48, 72, 96, 120, 144, 168]  # 1-7 days
        self.max_days = max_days
        self.max_hours = max_days * 24  # 168 hours for 7 days
        
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.raw_data = None
        self.processed_data = None
        self.account_stats = defaultdict(lambda: {'videos': 0, 'sequences': 0})
        
    def load_video_data(self) -> pd.DataFrame:
        """
        Load all video data from JSON files into a single DataFrame.
        
        Returns:
            DataFrame with all video snapshots
        """
        print("📂 Loading video data from JSON files...")
        all_records = []
        
        # Walk through all subdirectories to find video JSON files
        for json_file in self.video_data_dir.rglob("*videos_raw_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        for video in data:
                            record = self._extract_video_record(video)
                            if record:
                                all_records.append(record)
            except Exception as e:
                print(f"⚠️  Error loading {json_file}: {e}")
                continue
        
        print(f"✅ Loaded {len(all_records)} video snapshots")
        
        if not all_records:
            raise ValueError("No data loaded! Check your video_data_dir path.")
        
        df = pd.DataFrame(all_records)
        
        # Track accounts from the start
        initial_accounts = df['author_username'].unique()
        print(f"   Unique accounts in raw data: {len(initial_accounts)}")
        print(f"   Accounts: {sorted(initial_accounts)}")
        
        return df
    
    def _extract_video_record(self, video: Dict) -> Dict:
        """
        Extract required fields from a video JSON object.
        
        Args:
            video: Video dictionary from JSON
            
        Returns:
            Dictionary with standardized fields
        """
        try:
            stats = video.get('stats', {})
            
            # Parse timestamps
            create_time = video.get('create_time')
            if isinstance(create_time, str):
                posted_at = pd.to_datetime(create_time)
            else:
                # Use create_timestamp if available
                create_timestamp = video.get('create_timestamp')
                posted_at = pd.to_datetime(create_timestamp, unit='s')
            
            scraped_at = pd.to_datetime(video.get('scraped_at'))
            
            return {
                'video_id': str(video.get('video_id')),
                'author_username': video.get('author_username'),
                'posted_at': posted_at,
                'scraped_at': scraped_at,
                'views': stats.get('playCount', 0),
                'likes': stats.get('diggCount', 0),
                'shares': stats.get('shareCount', 0),
                'comments': stats.get('commentCount', 0),
                'collects': stats.get('collectCount', 0)
            }
        except Exception as e:
            print(f"⚠️  Error extracting record: {e}")
            return None
    
    def clean_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data with relaxed criteria and compute derived features.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame with derived features
        """
        print("\n🧹 Cleaning and preparing data (relaxed criteria)...")
        
        initial_accounts = set(df['author_username'].unique())
        initial_count = len(df)
        
        # Remove rows with missing critical fields
        df = df.dropna(subset=['video_id', 'posted_at', 'scraped_at', 'views'])
        
        # Remove negative or clearly broken values
        df = df[df['views'] >= 0]
        df = df[df['likes'] >= 0]
        df = df[df['shares'] >= 0]
        df = df[df['comments'] >= 0]
        
        # Calculate time since post (in hours)
        df['t_since_post'] = (df['scraped_at'] - df['posted_at']).dt.total_seconds() / 3600
        
        # Keep only snapshots within first 7 days (relaxed requirement)
        df = df[df['t_since_post'] <= self.max_hours]
        df = df[df['t_since_post'] >= 0]
        
        # Sort by video_id and scraped_at
        df = df.sort_values(['video_id', 'scraped_at'])
        
        # Remove duplicates (same video_id and scraped_at)
        df = df.drop_duplicates(subset=['video_id', 'scraped_at'], keep='first')
        
        # Check which accounts were lost during cleaning
        final_accounts = set(df['author_username'].unique())
        lost_accounts = initial_accounts - final_accounts
        
        print(f"✅ Data cleaned: {len(df)} records remaining (removed {initial_count - len(df)})")
        print(f"   Unique videos: {df['video_id'].nunique()}")
        print(f"   Unique accounts: {len(final_accounts)}")
        if lost_accounts:
            print(f"   ⚠️  Lost {len(lost_accounts)} accounts during cleaning: {sorted(lost_accounts)}")
        
        return df
    
    def prepare_snapshots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo chính xác 3 snapshots tại giờ 1, 2, 3 sau khi đăng.
        Dùng interpolation/extrapolation nếu không có snapshots đúng thời điểm.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame với 3 snapshots tại t=1h, 2h, 3h + temporal features
        """
        print(f"\n📸 Creating snapshots at t=1h, 2h, 3h for each video...")
        
        target_hours = [1, 2, 3]
        interpolated_dfs = []
        videos_kept = 0
        videos_skipped = 0
        stats = {'exact_match': 0, 'interpolated': 0, 'extrapolated': 0}
        
        for video_id, group in df.groupby('video_id'):
            group = group.sort_values('t_since_post')
            
            # Cần ít nhất 1 snapshot để có thể augment
            if len(group) < 1:
                videos_skipped += 1
                continue
            
            account = group['author_username'].iloc[0]
            posted_at = group['posted_at'].iloc[0]
            
            # Tạo 3 snapshots tại t=1, 2, 3
            snapshots_for_video = []
            
            for target_t in target_hours:
                snapshot = self._get_or_create_snapshot(group, target_t, posted_at, stats)
                if snapshot is not None:
                    snapshots_for_video.append(snapshot)
            
            # Chỉ giữ video nếu có đủ 3 snapshots
            if len(snapshots_for_video) == 3:
                video_df = pd.DataFrame(snapshots_for_video)
                interpolated_dfs.append(video_df)
                videos_kept += 1
                self.account_stats[account]['videos'] += 1
            else:
                videos_skipped += 1
        
        if not interpolated_dfs:
            raise ValueError("No videos could be prepared with 3 snapshots!")
        
        df_prepared = pd.concat(interpolated_dfs, ignore_index=True)
        
        print(f"✅ Snapshots prepared at t=1h, 2h, 3h:")
        print(f"   Videos kept: {videos_kept}")
        print(f"   Videos skipped: {videos_skipped}")
        print(f"   Exact matches: {stats['exact_match']}")
        print(f"   Interpolated: {stats['interpolated']}")
        print(f"   Extrapolated: {stats['extrapolated']}")
        print(f"   Total snapshots: {len(df_prepared)}")
        
        return df_prepared
    
    def _get_or_create_snapshot(self, group: pd.DataFrame, target_t: float, 
                                posted_at: pd.Timestamp, stats: dict) -> dict:
        """
        Lấy hoặc tạo snapshot tại thời điểm target_t (hours).
        Dùng interpolation/extrapolation với early growth consideration.
        
        Args:
            group: DataFrame of all snapshots for one video
            target_t: Target time in hours (1, 2, or 3)
            posted_at: Video posting timestamp
            stats: Dictionary to track statistics
            
        Returns:
            Dictionary representing the snapshot at target_t
        """
        # Tìm snapshot gần nhất
        time_diffs = np.abs(group['t_since_post'] - target_t)
        closest_idx = time_diffs.idxmin()
        closest_snapshot = group.loc[closest_idx]
        
        # Nếu có snapshot đủ gần (±30 phút), dùng trực tiếp
        if time_diffs[closest_idx] <= 0.5:
            stats['exact_match'] += 1
            snapshot = closest_snapshot.to_dict()
            snapshot['t_since_post'] = target_t
            snapshot['scraped_at'] = posted_at + timedelta(hours=float(target_t))
            snapshot['hour_of_day'] = snapshot['scraped_at'].hour
            snapshot['day_of_week'] = snapshot['scraped_at'].dayofweek
            return snapshot
        
        # Cần interpolate hoặc extrapolate
        before = group[group['t_since_post'] < target_t]
        after = group[group['t_since_post'] > target_t]
        
        if len(before) > 0 and len(after) > 0:
            # Interpolation: có snapshots trước và sau
            stats['interpolated'] += 1
            return self._interpolate_snapshot(before.iloc[-1], after.iloc[0], 
                                             target_t, posted_at)
        
        elif len(after) > 0:
            # Backward extrapolation: chỉ có snapshots sau
            # Dùng 2 snapshots đầu tiên để estimate growth rate
            stats['extrapolated'] += 1
            if len(after) >= 2:
                return self._extrapolate_backward(after.iloc[0], after.iloc[1], 
                                                  target_t, posted_at)
            else:
                # Chỉ có 1 snapshot sau, dùng flat extrapolation
                return self._flat_extrapolate(after.iloc[0], target_t, posted_at, direction='backward')
        
        elif len(before) > 0:
            # Forward extrapolation: chỉ có snapshots trước
            stats['extrapolated'] += 1
            if len(before) >= 2:
                return self._extrapolate_forward(before.iloc[-2], before.iloc[-1], 
                                                 target_t, posted_at)
            else:
                # Chỉ có 1 snapshot trước, dùng flat extrapolation
                return self._flat_extrapolate(before.iloc[0], target_t, posted_at, direction='forward')
        
        else:
            # Không thể tạo snapshot
            return None
    
    def _interpolate_snapshot(self, snap_before: pd.Series, snap_after: pd.Series, 
                              target_t: float, posted_at: pd.Timestamp) -> dict:
        """
        Interpolate metrics giữa 2 snapshots, với early growth boost.
        """
        t1 = snap_before['t_since_post']
        t2 = snap_after['t_since_post']
        
        # Linear interpolation weight
        alpha = (target_t - t1) / (t2 - t1) if t2 > t1 else 0.5
        
        # Early growth factor: trong 3h đầu, growth thường nhanh hơn
        # Nếu đang interpolate trong early hours, boost growth slightly
        early_factor = 1.0
        if target_t <= 3 and t1 <= 3:
            # Early hours có growth rate cao hơn 10-20%
            early_factor = 1.1 if target_t <= 2 else 1.05
        
        snapshot = snap_before.to_dict()
        
        # Interpolate metrics với early growth consideration
        for col in ['views', 'likes', 'shares', 'comments', 'collects']:
            val1 = snap_before[col]
            val2 = snap_after[col]
            
            # Linear interpolation với early boost
            interpolated = val1 + alpha * (val2 - val1) * early_factor
            snapshot[col] = max(0, int(interpolated))
        
        # Update time fields
        snapshot['t_since_post'] = target_t
        snapshot['scraped_at'] = posted_at + timedelta(hours=float(target_t))
        snapshot['hour_of_day'] = snapshot['scraped_at'].hour
        snapshot['day_of_week'] = snapshot['scraped_at'].dayofweek
        
        return snapshot
    
    def _extrapolate_backward(self, snap1: pd.Series, snap2: pd.Series, 
                              target_t: float, posted_at: pd.Timestamp) -> dict:
        """
        Extrapolate backward từ 2 snapshots sau đó về target_t.
        Giả sử growth rate ở đầu nhanh hơn.
        """
        t1 = snap1['t_since_post']
        t2 = snap2['t_since_post']
        
        # Tính average growth rate giữa snap1 và snap2
        time_diff = t2 - t1
        
        snapshot = snap1.to_dict()
        
        for col in ['views', 'likes', 'shares', 'comments', 'collects']:
            val1 = snap1[col]
            val2 = snap2[col]
            
            if time_diff > 0 and val1 > 0:
                # Growth rate per hour
                growth_rate = (val2 - val1) / time_diff
                
                # Early boost: ở giờ đầu, growth thường nhanh hơn 20-30%
                early_boost = 1.25 if target_t <= 2 else 1.15
                adjusted_growth = growth_rate * early_boost
                
                # Extrapolate backward
                time_back = snap1['t_since_post'] - target_t
                estimated_val = val1 - adjusted_growth * time_back
                
                # Ensure non-negative và reasonable
                snapshot[col] = max(0, int(estimated_val))
            else:
                # Không thể estimate, dùng flat
                snapshot[col] = val1
        
        snapshot['t_since_post'] = target_t
        snapshot['scraped_at'] = posted_at + timedelta(hours=float(target_t))
        snapshot['hour_of_day'] = snapshot['scraped_at'].hour
        snapshot['day_of_week'] = snapshot['scraped_at'].dayofweek
        
        return snapshot
    
    def _extrapolate_forward(self, snap1: pd.Series, snap2: pd.Series, 
                             target_t: float, posted_at: pd.Timestamp) -> dict:
        """
        Extrapolate forward từ 2 snapshots trước đó ra target_t.
        Giả sử growth rate giảm dần theo thời gian.
        """
        t1 = snap1['t_since_post']
        t2 = snap2['t_since_post']
        
        time_diff = t2 - t1
        
        snapshot = snap2.to_dict()
        
        for col in ['views', 'likes', 'shares', 'comments', 'collects']:
            val1 = snap1[col]
            val2 = snap2[col]
            
            if time_diff > 0:
                # Growth rate per hour
                growth_rate = (val2 - val1) / time_diff
                
                # Decay factor: growth rate giảm dần
                time_forward = target_t - snap2['t_since_post']
                decay_factor = 0.9  # Growth chậm hơn 10% khi extrapolate
                adjusted_growth = growth_rate * decay_factor
                
                estimated_val = val2 + adjusted_growth * time_forward
                snapshot[col] = max(val2, int(estimated_val))  # Ensure monotonic
            else:
                snapshot[col] = val2
        
        snapshot['t_since_post'] = target_t
        snapshot['scraped_at'] = posted_at + timedelta(hours=float(target_t))
        snapshot['hour_of_day'] = snapshot['scraped_at'].hour
        snapshot['day_of_week'] = snapshot['scraped_at'].dayofweek
        
        return snapshot
    
    def _flat_extrapolate(self, snap: pd.Series, target_t: float, 
                          posted_at: pd.Timestamp, direction: str) -> dict:
        """
        Flat extrapolation khi chỉ có 1 snapshot.
        """
        snapshot = snap.to_dict()
        
        # Giữ nguyên các metrics (conservative approach)
        # Có thể adjust nhẹ dựa trên direction
        if direction == 'backward':
            # Backward: giảm nhẹ metrics (80-90%)
            for col in ['views', 'likes', 'shares', 'comments', 'collects']:
                snapshot[col] = int(snap[col] * 0.85)
        # Forward: giữ nguyên (conservative)
        
        snapshot['t_since_post'] = target_t
        snapshot['scraped_at'] = posted_at + timedelta(hours=float(target_t))
        snapshot['hour_of_day'] = snapshot['scraped_at'].hour
        snapshot['day_of_week'] = snapshot['scraped_at'].dayofweek
        
        return snapshot
    
    def resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample each video to hourly intervals (0-168 hours) with interpolation for missing hours.
        
        Args:
            df: Cleaned DataFrame with irregular snapshots
            
        Returns:
            DataFrame with hourly snapshots for each video
        """
        print(f"\n⏰ Resampling videos to hourly intervals (0-{self.max_hours}h)...")
        
        resampled_dfs = []
        videos_processed = 0
        videos_skipped = 0
        
        for video_id, group in df.groupby('video_id'):
            group = group.sort_values('t_since_post')
            
            # Need at least 2 snapshots to interpolate
            if len(group) < 2:
                videos_skipped += 1
                continue
            
            # Get account and posted_at
            account = group['author_username'].iloc[0]
            posted_at = group['posted_at'].iloc[0]
            
            # Create hourly timeline from 0 to max_hours
            hourly_timeline = np.arange(0, self.max_hours + 1, 1)  # 0, 1, 2, ..., 168
            
            # Interpolate metrics for each hour
            hourly_data = []
            for hour in hourly_timeline:
                # Find closest snapshots before and after this hour
                before = group[group['t_since_post'] <= hour]
                after = group[group['t_since_post'] >= hour]
                
                if len(before) == 0:
                    # Use first available snapshot
                    snapshot = group.iloc[0].copy()
                elif len(after) == 0:
                    # Use last available snapshot (flat extrapolation)
                    snapshot = group.iloc[-1].copy()
                elif before.iloc[-1]['t_since_post'] == hour:
                    # Exact match exists
                    snapshot = before.iloc[-1].copy()
                else:
                    # Interpolate between before and after
                    snap_before = before.iloc[-1]
                    snap_after = after.iloc[0]
                    
                    t1 = snap_before['t_since_post']
                    t2 = snap_after['t_since_post']
                    alpha = (hour - t1) / (t2 - t1) if t2 > t1 else 0
                    
                    # Linear interpolation for metrics
                    snapshot = snap_before.copy()
                    for col in ['views', 'likes', 'shares', 'comments', 'collects']:
                        val1 = snap_before[col]
                        val2 = snap_after[col]
                        snapshot[col] = int(val1 + alpha * (val2 - val1))
                
                # Update time fields
                snapshot['t_since_post'] = hour
                snapshot['scraped_at'] = posted_at + timedelta(hours=float(hour))
                
                hourly_data.append(snapshot)
            
            # Create DataFrame for this video
            video_hourly = pd.DataFrame(hourly_data)
            resampled_dfs.append(video_hourly)
            videos_processed += 1
            
            # Track for account stats
            self.account_stats[account]['videos'] += 1
        
        df_resampled = pd.concat(resampled_dfs, ignore_index=True)
        
        print(f"✅ Resampling complete:")
        print(f"   Videos processed: {videos_processed}")
        print(f"   Videos skipped (insufficient snapshots): {videos_skipped}")
        print(f"   Total hourly snapshots: {len(df_resampled)}")
        print(f"   Expected per video: {self.max_hours + 1} hours")
        
        return df_resampled
    
    def augment_videos_for_accounts(self, df: pd.DataFrame, min_videos_base: int = 30) -> pd.DataFrame:
        """
        Create synthetic videos for accounts with insufficient data using interpolation.
        This helps ensure all accounts have enough data for per-KOL training (MVP approach).
        Target: 30 + random(0-10) videos per account for robust train/val/test splits.
        
        Args:
            df: Cleaned DataFrame
            min_videos_base: Base minimum videos per account (default: 30)
            
        Returns:
            DataFrame with augmented synthetic videos
        """
        print(f"\n🔬 Augmenting data for accounts with <{min_videos_base} videos...")
        
        augmented_dfs = []
        augmentation_stats = {}
        
        for account, account_group in df.groupby('author_username'):
            n_videos = account_group['video_id'].nunique()
            
            # Add small random number to target (30 + 0-10 = 30-40 videos per account)
            min_videos_target = min_videos_base + np.random.randint(0, 11)
            
            if n_videos >= min_videos_target:
                # Account has enough videos, no augmentation needed
                augmented_dfs.append(account_group)
                augmentation_stats[account] = {'original': n_videos, 'synthetic': 0, 'total': n_videos, 'target': min_videos_target}
                continue
            
            # Need to create synthetic videos
            n_synthetic_needed = min_videos_target - n_videos
            augmented_dfs.append(account_group)
            
            # Get existing videos for this account
            existing_videos = account_group.groupby('video_id')
            video_list = list(existing_videos)
            
            if len(video_list) < 2:
                # Can't interpolate with only 1 video, just duplicate it
                for i in range(n_synthetic_needed):
                    synthetic_video = account_group.copy()
                    synthetic_video['video_id'] = f"{account_group['video_id'].iloc[0]}_synthetic_{i+1}"
                    synthetic_video['is_synthetic'] = True
                    augmented_dfs.append(synthetic_video)
            else:
                # Interpolate between pairs of videos
                for i in range(n_synthetic_needed):
                    # Pick two random videos to interpolate between
                    idx1, idx2 = np.random.choice(len(video_list), 2, replace=False)
                    vid1_id, vid1_data = video_list[idx1]
                    vid2_id, vid2_data = video_list[idx2]
                    
                    # Create synthetic video by interpolating
                    synthetic_video = self._interpolate_videos(vid1_data, vid2_data, account, i+1)
                    augmented_dfs.append(synthetic_video)
            
            augmentation_stats[account] = {
                'original': n_videos, 
                'synthetic': n_synthetic_needed, 
                'total': n_videos + n_synthetic_needed,
                'target': min_videos_target
            }
        
        df_augmented = pd.concat(augmented_dfs, ignore_index=True)
        
        # Count augmented accounts
        augmented_count = sum(1 for stats in augmentation_stats.values() if stats['synthetic'] > 0)
        total_synthetic = sum(stats['synthetic'] for stats in augmentation_stats.values())
        
        print(f"✅ Augmentation complete:")
        print(f"   Accounts augmented: {augmented_count}")
        print(f"   Synthetic videos created: {total_synthetic}")
        print(f"   Total videos now: {df_augmented['video_id'].nunique()}")
        
        # Store stats for reporting
        self.augmentation_stats = augmentation_stats
        
        return df_augmented
    
    def _interpolate_videos(self, vid1: pd.DataFrame, vid2: pd.DataFrame, account: str, idx: int) -> pd.DataFrame:
        """
        Create a synthetic video by interpolating between two real videos.
        
        Args:
            vid1, vid2: DataFrames of two videos to interpolate between
            account: Account name
            idx: Index for synthetic video naming
            
        Returns:
            DataFrame representing synthetic video
        """
        vid1 = vid1.sort_values('scraped_at')
        vid2 = vid2.sort_values('scraped_at')
        
        # Use the video with more snapshots as template
        template = vid1 if len(vid1) >= len(vid2) else vid2
        synthetic = template.copy()
        
        # Generate new video_id
        synthetic['video_id'] = f"{account}_synthetic_video_{idx}"
        synthetic['is_synthetic'] = True
        
        # Interpolate numeric metrics (blend between the two videos)
        alpha = np.random.uniform(0.3, 0.7)  # Interpolation weight
        
        for col in ['views', 'likes', 'shares', 'comments', 'collects']:
            if col in vid1.columns and col in vid2.columns:
                # Match by index position (rough time alignment)
                for i in range(min(len(synthetic), len(vid1), len(vid2))):
                    val1 = vid1.iloc[i][col] if i < len(vid1) else vid1.iloc[-1][col]
                    val2 = vid2.iloc[i][col] if i < len(vid2) else vid2.iloc[-1][col]
                    synthetic.iloc[i, synthetic.columns.get_loc(col)] = int(alpha * val1 + (1 - alpha) * val2)
        
        # Add small random noise to make it more realistic (±5%)
        for col in ['views', 'likes', 'shares', 'comments', 'collects']:
            if col in synthetic.columns:
                noise = np.random.uniform(0.95, 1.05, len(synthetic))
                synthetic[col] = (synthetic[col] * noise).astype(int)
                synthetic[col] = synthetic[col].clip(lower=0)  # Ensure non-negative
        
        # Ensure monotonically increasing values (views/engagement should never decrease)
        for col in ['views', 'likes', 'shares', 'comments', 'collects']:
            if col in synthetic.columns:
                # Sort by scraped_at to ensure temporal order
                synthetic = synthetic.sort_values('scraped_at')
                values = synthetic[col].values
                
                # Force monotonic increase: each value must be >= previous value
                for i in range(1, len(values)):
                    if values[i] < values[i-1]:
                        values[i] = values[i-1]
                
                synthetic[col] = values
        
        # Cap unrealistic growth rates (prevent 300k-400k jumps in 30min-1hour)
        # Max realistic growth rates per hour: views=150k, likes=15k, shares=5k, comments=2k
        max_growth_rates = {
            'views': 150000,      # Max 150k views/hour (viral videos)
            'likes': 15000,       # Max 15k likes/hour
            'shares': 5000,       # Max 5k shares/hour  
            'comments': 2000,     # Max 2k comments/hour
            'collects': 3000      # Max 3k collects/hour
        }
        
        synthetic = synthetic.sort_values('scraped_at')
        
        for col in ['views', 'likes', 'shares', 'comments', 'collects']:
            if col in synthetic.columns and 't_since_post' in synthetic.columns:
                values = synthetic[col].values.copy()
                times = synthetic['t_since_post'].values  # In hours
                
                # Check and cap growth between consecutive snapshots
                for i in range(1, len(values)):
                    time_diff = max(times[i] - times[i-1], 0.01)  # Avoid division by zero
                    growth = values[i] - values[i-1]
                    growth_rate = growth / time_diff  # Growth per hour
                    
                    # If growth rate exceeds max, cap the value
                    if growth_rate > max_growth_rates[col]:
                        max_allowed_growth = max_growth_rates[col] * time_diff
                        values[i] = int(values[i-1] + max_allowed_growth)
                
                synthetic[col] = values
        
        return synthetic
    
    
    def filter_videos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out accounts without recent videos.
        Removes videos/accounts that don't have data within the specified time window.
        
        Args:
            df: DataFrame after resampling
            
        Returns:
            Filtered DataFrame
        """
        print(f"\n⏱️  Filtering videos (must have data within last {self.max_days} days)...")
        
        initial_accounts = set(df['author_username'].unique())
        initial_videos = df['video_id'].nunique()
        
        # Keep only accounts that have videos posted recently
        # (This is already handled by max_hours filtering in clean_and_prepare,
        # but we can add additional account-level filtering here if needed)
        
        filtered_groups = []
        for video_id, group in df.groupby('video_id'):
            # Video already resampled to hourly, so it has sufficient data
            filtered_groups.append(group)
            
            # Track account stats
            account = group['author_username'].iloc[0]
            self.account_stats[account]['videos'] += 1
        
        if not filtered_groups:
            raise ValueError("No videos remaining after filtering!")
        
        df_filtered = pd.concat(filtered_groups, ignore_index=True)
        
        # Check which accounts lost all their videos
        final_accounts = set(df_filtered['author_username'].unique())
        lost_accounts = initial_accounts - final_accounts
        
        print(f"✅ Kept {len(filtered_groups)} videos")
        print(f"   Total hourly records: {len(df_filtered)}")
        print(f"   Unique accounts: {len(final_accounts)}")
        if lost_accounts:
            print(f"   ⚠️  Lost {len(lost_accounts)} accounts: {sorted(lost_accounts)}")
        
        return df_filtered
        
        if lost_accounts:
            print(f"   ⚠️  Lost {len(lost_accounts)} accounts (all videos had <{self.min_snapshots} snapshots):")
            for acc in sorted(lost_accounts):
                print(f"      • {acc}: {dropped_accounts[acc]['videos']} videos dropped")
        
        # Show time resolution stats
        time_diffs = []
        for vid, grp in df_filtered.groupby('video_id'):
            grp = grp.sort_values('scraped_at')
            diffs = grp['scraped_at'].diff().dt.total_seconds() / 60  # minutes
            time_diffs.extend(diffs.dropna().tolist())
        
        if time_diffs:
            print(f"   Time resolution: {np.median(time_diffs):.0f}min median (range: {min(time_diffs):.0f}-{max(time_diffs):.0f}min)")
        
        return df_filtered
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Định nghĩa feature columns bao gồm cả target_horizon.
        Temporal features (hour_of_day, day_of_week) đã được thêm trong prepare_snapshots.
        
        Args:
            df: DataFrame with 3 snapshots per video per horizon
            
        Returns:
            DataFrame with feature columns defined
        """
        print("\n🔧 Setting feature columns...")
        
        # 4 metrics + temporal features + target_horizon
        self.feature_columns = [
            'views', 
            'likes', 
            'shares', 
            'comments',
            'hour_of_day',
            'day_of_week',
            'target_horizon'
        ]
        
        print(f"✅ Features: {', '.join(self.feature_columns)}")
        
        return df
    
    def _compute_velocity(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute growth velocity features (consistent 1-hour intervals)."""
        group = group.sort_values('scraped_at')
        
        # With hourly resampling, velocity is simply the difference (views per hour)
        group['views_velocity'] = group['views'].diff().fillna(0)
        group['likes_velocity'] = group['likes'].diff().fillna(0)
        group['shares_velocity'] = group['shares'].diff().fillna(0)
        
        # Clip negative velocities (shouldn't happen with monotonic data, but just in case)
        group['views_velocity'] = group['views_velocity'].clip(lower=0)
        group['likes_velocity'] = group['likes_velocity'].clip(lower=0)
        group['shares_velocity'] = group['shares_velocity'].clip(lower=0)
        
        return group
    
    def expand_with_horizons(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand dataframe: mỗi video (3 snapshots) → 7 nhóm (mỗi horizon 1 nhóm).
        Mỗi nhóm có: 3 snapshots + target_horizon + target metrics cho horizon đó.
        IMPORTANT: Ensures monotonic increase - bigger horizons must have bigger metrics.
        
        Args:
            df: DataFrame with 3 snapshots per video
            
        Returns:
            Expanded DataFrame with target_horizon and target metrics columns
        """
        print(f"\n🔄 Expanding data with time horizons...")
        
        # Prepare raw data for target lookup
        raw_df = self.raw_data.copy()
        raw_df['t_since_post'] = (raw_df['scraped_at'] - raw_df['posted_at']).dt.total_seconds() / 3600
        
        expanded_dfs = []
        horizon_stats = {T: 0 for T in self.prediction_horizons}
        skipped_monotonic = 0
        
        for video_id, group in df.groupby('video_id'):
            if len(group) != 3:
                continue
            
            video_raw = raw_df[raw_df['video_id'] == video_id]
            
            # First, collect all targets for this video
            video_targets = {}
            for T in self.prediction_horizons:
                target_snapshots = video_raw[
                    (video_raw['t_since_post'] >= T - 2) &
                    (video_raw['t_since_post'] <= T + 2)
                ]
                
                if len(target_snapshots) == 0:
                    continue
                
                closest_idx = (target_snapshots['t_since_post'] - T).abs().idxmin()
                target_snap = target_snapshots.loc[closest_idx]
                
                video_targets[T] = {
                    'views': target_snap['views'],
                    'likes': target_snap['likes'],
                    'shares': target_snap['shares'],
                    'comments': target_snap['comments']
                }
            
            # Check monotonic constraint
            sorted_horizons = sorted(video_targets.keys())
            is_monotonic = True
            
            for i in range(1, len(sorted_horizons)):
                prev_T = sorted_horizons[i-1]
                curr_T = sorted_horizons[i]
                
                # All metrics must be non-decreasing
                for metric in ['views', 'likes', 'shares', 'comments']:
                    if video_targets[curr_T][metric] < video_targets[prev_T][metric]:
                        is_monotonic = False
                        break
                
                if not is_monotonic:
                    break
            
            if not is_monotonic:
                skipped_monotonic += 1
                continue
            
            # Now create expanded records for each horizon
            for T in sorted_horizons:
                group_copy = group.copy()
                group_copy['target_horizon'] = T
                group_copy['target_views'] = video_targets[T]['views']
                group_copy['target_likes'] = video_targets[T]['likes']
                group_copy['target_shares'] = video_targets[T]['shares']
                group_copy['target_comments'] = video_targets[T]['comments']
                
                expanded_dfs.append(group_copy)
                horizon_stats[T] += 1
        
        if not expanded_dfs:
            raise ValueError("No videos have targets for any horizon!")
        
        df_expanded = pd.concat(expanded_dfs, ignore_index=True)
        
        print(f"✅ Expansion complete:")
        print(f"   Total records: {len(df_expanded)} (3 snapshots × videos × horizons)")
        print(f"   Videos skipped (monotonic violation): {skipped_monotonic}")
        print(f"   Videos with targets per horizon:")
        for T in self.prediction_horizons:
            print(f"      T={T}h ({T//24}d): {horizon_stats[T]} videos")
        
        return df_expanded
    
    def create_targets(self, df: pd.DataFrame, prediction_horizon: int) -> pd.DataFrame:
        """
        Create prediction targets T hours ahead for multiple metrics.
        
        Args:
            df: DataFrame with features
            prediction_horizon: Hours ahead to predict (T in X+T)
            
        Returns:
            DataFrame with target columns for this specific horizon
        """
        # Initialize target columns
        df[f'target_views_{prediction_horizon}h'] = np.nan
        df[f'target_likes_{prediction_horizon}h'] = np.nan
        df[f'target_shares_{prediction_horizon}h'] = np.nan
        df[f'target_comments_{prediction_horizon}h'] = np.nan
        
        for video_id, group in df.groupby('video_id'):
            group = group.sort_values('hours_since_post')
            
            # For each hour, find the value T hours later
            for idx in group.index:
                current_hour = group.loc[idx, 'hours_since_post']
                target_hour = current_hour + prediction_horizon
                
                # Find the row at target_hour
                target_rows = group[group['hours_since_post'] == target_hour]
                
                if len(target_rows) > 0:
                    target_row = target_rows.iloc[0]
                    df.at[idx, f'target_views_{prediction_horizon}h'] = target_row['views']
                    df.at[idx, f'target_likes_{prediction_horizon}h'] = target_row['likes']
                    df.at[idx, f'target_shares_{prediction_horizon}h'] = target_row['shares']
                    df.at[idx, f'target_comments_{prediction_horizon}h'] = target_row['comments']
        
        return df
    
    def split_data(self, df: pd.DataFrame, 
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by video_id WITHIN each KOL to ensure each account has train/val/test data.
        This is critical for per-KOL model training.
        
        Args:
            df: Full DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"\n✂️  Splitting data by videos WITHIN each KOL (train: {train_ratio}, val: {val_ratio}, test: {1-train_ratio-val_ratio})...")
        
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        np.random.seed(42)  # For reproducibility
        
        for account, account_group in df.groupby('author_username'):
            # Get unique video IDs for this account
            video_ids = account_group['video_id'].unique()
            n_videos = len(video_ids)
            
            if n_videos < 3:
                # If account has fewer than 3 videos, put all in train
                train_dfs.append(account_group)
                print(f"   {account}: {n_videos} videos (all → train, insufficient for split)")
                continue
            
            # Shuffle videos for this account
            shuffled_videos = video_ids.copy()
            np.random.shuffle(shuffled_videos)
            
            # Split video IDs for this account
            train_end = max(1, int(n_videos * train_ratio))
            val_end = max(train_end + 1, int(n_videos * (train_ratio + val_ratio)))
            
            train_videos = shuffled_videos[:train_end]
            val_videos = shuffled_videos[train_end:val_end]
            test_videos = shuffled_videos[val_end:]
            
            # Split data based on video IDs
            account_train = account_group[account_group['video_id'].isin(train_videos)]
            account_val = account_group[account_group['video_id'].isin(val_videos)]
            account_test = account_group[account_group['video_id'].isin(test_videos)]
            
            train_dfs.append(account_train)
            if len(account_val) > 0:
                val_dfs.append(account_val)
            if len(account_test) > 0:
                test_dfs.append(account_test)
            
            print(f"   {account}: {n_videos} videos → train:{len(train_videos)}, val:{len(val_videos)}, test:{len(test_videos)}")
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
        test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
        
        print(f"\n✅ Split complete:")
        print(f"   Train: {len(train_df)} records from {train_df['video_id'].nunique()} videos, {train_df['author_username'].nunique()} accounts")
        print(f"   Val:   {len(val_df)} records from {val_df['video_id'].nunique() if len(val_df) > 0 else 0} videos, {val_df['author_username'].nunique() if len(val_df) > 0 else 0} accounts")
        print(f"   Test:  {len(test_df)} records from {test_df['video_id'].nunique() if len(test_df) > 0 else 0} videos, {test_df['author_username'].nunique() if len(test_df) > 0 else 0} accounts")
        
        return train_df, val_df, test_df
    
    def scale_features(self, train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale all features including target_horizon.
        
        Args:
            train_df, val_df, test_df: Split DataFrames
            
        Returns:
            Tuple of scaled DataFrames
        """
        print("\n📊 Scaling features...")
        
        # Scale all features from feature_columns
        scale_columns = self.feature_columns
        
        if len(train_df) == 0:
            raise ValueError("Train set is empty!")
        
        # Fit scaler on training data only
        self.scaler.fit(train_df[scale_columns])
        
        # Transform all splits
        train_df_scaled = train_df.copy()
        train_df_scaled[scale_columns] = self.scaler.transform(train_df[scale_columns])
        
        val_df_scaled = val_df.copy() if len(val_df) > 0 else pd.DataFrame()
        if len(val_df) > 0:
            val_df_scaled[scale_columns] = self.scaler.transform(val_df[scale_columns])
        
        test_df_scaled = test_df.copy() if len(test_df) > 0 else pd.DataFrame()
        if len(test_df) > 0:
            test_df_scaled[scale_columns] = self.scaler.transform(test_df[scale_columns])
        
        print(f"✅ Scaled {len(scale_columns)} features: {', '.join(scale_columns)}")
        
        return train_df_scaled, val_df_scaled, test_df_scaled
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[int]]:
        """
        Tạo sequences từ expanded dataframe:
        - Group by (video_id, target_horizon) → lấy 3 snapshots
        - Input X: 3 snapshots × 7 features (4 metrics + hour_of_day + day_of_week + target_horizon)
        - Output y: 4 target metrics (target_views, target_likes, target_shares, target_comments)
        
        Returns:
            X: shape (N, 3, 7)
            y: shape (N, 4)
        """
        print(f"\n🪟 Creating sequences from expanded data...")
        
        X_list = []
        y_list = []
        video_id_list = []
        account_list = []
        horizon_list = []
        
        videos_with_sequences = set()
        horizon_stats = {T: 0 for T in self.prediction_horizons}
        
        # Group by (video_id, target_horizon)
        for (video_id, target_horizon), group in df.groupby(['video_id', 'target_horizon']):
            group = group.sort_values('scraped_at')
            
            # Phải có đúng 3 snapshots
            if len(group) != 3:
                continue
            
            account = group['author_username'].iloc[0]
            
            # X: 3 snapshots × 7 features
            X_sequence = []
            for idx in range(3):
                snap = group.iloc[idx]
                features = [
                    snap['views'],
                    snap['likes'],
                    snap['shares'],
                    snap['comments'],
                    snap['hour_of_day'],
                    snap['day_of_week'],
                    snap['target_horizon']
                ]
                X_sequence.append(features)
            
            # y: target metrics (same for all 3 snapshots in this group)
            snap = group.iloc[0]
            y_values = [
                snap['target_views'],
                snap['target_likes'],
                snap['target_shares'],
                snap['target_comments']
            ]
            
            X_list.append(X_sequence)
            y_list.append(y_values)
            video_id_list.append(video_id)
            account_list.append(account)
            horizon_list.append(target_horizon)  # Keep as-is (may be scaled)
            
            videos_with_sequences.add(video_id)
            # For stats, need original horizon value - get from unscaled df
            # Just count all for now
            if target_horizon not in horizon_stats:
                horizon_stats[target_horizon] = 0
            horizon_stats[target_horizon] += 1
        
        # Count sequences per account properly
        for account in df['author_username'].unique():
            account_seqs = sum(1 for a in account_list if a == account)
            self.account_stats[account]['sequences'] = account_seqs
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"✅ Created {len(X)} sequences")
        if len(X) > 0:
            print(f"   X shape: {X.shape} - (samples, 3 snapshots, 7 features)")
            print(f"   y shape: {y.shape} - (samples, 4 metrics)")
            print(f"   Features: [views, likes, shares, comments, hour_of_day, day_of_week, target_horizon]")
            print(f"\n   Sequences per horizon:")
            for T in self.prediction_horizons:
                if T in horizon_stats:
                    print(f"      T={T}h ({T//24}d): {horizon_stats[T]} sequences")
        
        return X, y, video_id_list, account_list, horizon_list
    
    def save_artifacts(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                       train_accounts: List[str], val_accounts: List[str], test_accounts: List[str]):
        """Save all artifacts for later use, including account statistics."""
        print("\n💾 Saving artifacts...")
        
        # Save DataFrames
        train_df.to_csv(self.output_dir / 'train_data.csv', index=False)
        val_df.to_csv(self.output_dir / 'val_data.csv', index=False)
        test_df.to_csv(self.output_dir / 'test_data.csv', index=False)
        
        # Save scaler
        with open(self.output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Count sequences per account per split
        from collections import Counter
        train_account_seqs = Counter(train_accounts)
        val_account_seqs = Counter(val_accounts)
        test_account_seqs = Counter(test_accounts)
        
        # Get all accounts (including those with 0 sequences)
        all_accounts = set(self.account_stats.keys())
        
        account_summary = {}
        for account in sorted(all_accounts):
            account_summary[account] = {
                'total_videos': self.account_stats[account]['videos'],
                'total_sequences': self.account_stats[account]['sequences'],
                'train_sequences': train_account_seqs.get(account, 0),
                'val_sequences': val_account_seqs.get(account, 0),
                'test_sequences': test_account_seqs.get(account, 0)
            }
        
        # Save metadata with account stats and augmentation info
        metadata = {
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'prediction_horizons': self.prediction_horizons,
            'max_days': self.max_days,
            'max_hours': self.max_hours,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'input_description': '3 snapshots (hour 1,2,3) with 4 metrics + temporal features + time_horizon',
            'input_shape': '(n_samples, 3, 7) - 3 snapshots x 7 features',
            'input_features': 'views, likes, shares, comments, hour_of_day, day_of_week, time_horizon',
            'target_columns': ['views', 'likes', 'shares', 'comments'],
            'target_description': 'Metrics at specified time_horizon (1-7 days)',
            'target_shape': '(n_samples, 4)',
            'time_horizons': self.prediction_horizons,
            'time_horizons_days': [h//24 for h in self.prediction_horizons],
            'model_type': 'LSTM supervised learning - predict metrics at 1-7 days',
            'account_statistics': account_summary,
            'augmentation_applied': hasattr(self, 'augmentation_stats'),
            'augmentation_stats': getattr(self, 'augmentation_stats', {}),
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Artifacts saved to {self.output_dir}")
        
        # Generate report with augmentation info
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("LSTM DATA PROCESSING REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"")
        report_lines.append(f"Configuration:")
        report_lines.append(f"  - Sequence length: {self.sequence_length}")
        report_lines.append(f"  - Prediction horizons: {self.prediction_horizons}")
        report_lines.append(f"  - Model type: Variable horizon (single model predicts any T)")
        report_lines.append(f"  - Max days coverage: {self.max_days}")
        report_lines.append(f"  - Data augmentation: {'ENABLED' if hasattr(self, 'augmentation_stats') else 'DISABLED'}")
        report_lines.append(f"")
        
        # Augmentation summary
        if hasattr(self, 'augmentation_stats'):
            augmented = [acc for acc, stats in self.augmentation_stats.items() if stats['synthetic'] > 0]
            total_synthetic = sum(stats['synthetic'] for stats in self.augmentation_stats.values())
            
            report_lines.append(f"Data Augmentation Summary:")
            report_lines.append(f"  - Accounts augmented: {len(augmented)}/{len(self.augmentation_stats)}")
            report_lines.append(f"  - Synthetic videos created: {total_synthetic}")
            report_lines.append(f"  - Method: Linear interpolation between existing videos + noise")
            report_lines.append(f"")
            report_lines.append(f"  Augmented accounts:")
            for acc in sorted(augmented):
                stats = self.augmentation_stats[acc]
                report_lines.append(f"    • {acc}: {stats['original']} real + {stats['synthetic']} synthetic = {stats['total']} total")
            report_lines.append(f"")
        
        report_lines.append(f"Sequence counts per account:")
        report_lines.append(f"{'Account':<30} {'Videos':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
        report_lines.append("-" * 80)
        
        total_videos = 0
        total_train = 0
        total_val = 0
        total_test = 0
        total_seqs = 0
        
        for account in sorted(all_accounts):
            stats = account_summary[account]
            line = f"{account:<30} {stats['total_videos']:<10} " \
                   f"{stats['train_sequences']:<10} {stats['val_sequences']:<10} " \
                   f"{stats['test_sequences']:<10} {stats['total_sequences']:<10}"
            report_lines.append(line)
            
            total_videos += stats['total_videos']
            total_train += stats['train_sequences']
            total_val += stats['val_sequences']
            total_test += stats['test_sequences']
            total_seqs += stats['total_sequences']
        
        report_lines.append("-" * 80)
        report_lines.append(f"{'TOTAL':<30} {total_videos:<10} {total_train:<10} {total_val:<10} {total_test:<10} {total_seqs:<10}")
        report_lines.append("")
        report_lines.append("="*80)
        
        # Print to console
        print(f"\n📊 Report Preview:")
        for line in report_lines:
            print(line)
        
        # Write to file
        report_path = self.output_dir / 'process_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n📄 Full report saved to: {report_path}")
    
    def run_pipeline(self):
        """Run the complete data preparation pipeline."""
        print("=" * 60)
        print("🚀 LSTM Data Preparation - Theo yêu cầu thầy")
        print("=" * 60)
        print(f"📊 Model Input: 3 snapshots (giờ 1, 2, 3)")
        print(f"   - Mỗi snapshot: views, likes, shares, comments, hour_of_day, day_of_week")
        print(f"   - Thêm: time_horizon (1-7 days)")
        print(f"📊 Model Output: views, likes, shares, comments tại time_horizon")
        horizons_str = ', '.join([f"{h}h ({h//24}d)" for h in self.prediction_horizons])
        print(f"⏰ Time horizons: {horizons_str}")
        print("=" * 60)
        
        # 1. Load data
        self.raw_data = self.load_video_data()
        
        # 2. Clean and prepare
        df = self.clean_and_prepare(self.raw_data)
        
        # 3. Keep only first 3 snapshots per video + add temporal features
        df = self.prepare_snapshots(df)
        
        # 4. Augment data (synthetic videos)
        df = self.augment_videos_for_accounts(df, min_videos_base=30)
        
        # 5. Expand with time horizons and find targets
        df = self.expand_with_horizons(df)
        
        # 6. Define feature columns (including target_horizon)
        df = self.create_features(df)
        
        # 7. Split data by video
        train_df, val_df, test_df = self.split_data(df)
        
        # 8. Scale features (including target_horizon)
        train_df_scaled, val_df_scaled, test_df_scaled = self.scale_features(
            train_df, val_df, test_df
        )
        
        # 9. Create sequences from expanded data
        X_train, y_train, train_video_ids, train_accounts, train_horizons = self.create_sequences(train_df_scaled)
        X_val, y_val, val_video_ids, val_accounts, val_horizons = self.create_sequences(val_df_scaled)
        X_test, y_test, test_video_ids, test_accounts, test_horizons = self.create_sequences(test_df_scaled)
        
        # 10. Save artifacts (including account stats)
        self.save_artifacts(train_df_scaled, val_df_scaled, test_df_scaled,
                           train_accounts, val_accounts, test_accounts)
        
        # Save sequences and targets
        np.save(self.output_dir / 'X_train.npy', X_train)
        np.save(self.output_dir / 'X_val.npy', X_val)
        np.save(self.output_dir / 'X_test.npy', X_test)
        
        np.save(self.output_dir / 'y_train.npy', y_train)
        np.save(self.output_dir / 'y_val.npy', y_val)
        np.save(self.output_dir / 'y_test.npy', y_test)
        
        # Save video/account IDs
        with open(self.output_dir / 'sequence_info.json', 'w') as f:
            json.dump({
                'train': {'video_ids': train_video_ids, 'accounts': train_accounts},
                'val': {'video_ids': val_video_ids, 'accounts': val_accounts},
                'test': {'video_ids': test_video_ids, 'accounts': test_accounts}
            }, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✅ Pipeline Complete!")
        print("=" * 60)
        print(f"\n📁 Output directory: {self.output_dir.absolute()}")
        print(f"\n📊 Summary:")
        print(f"   Train sequences: {len(X_train)}")
        print(f"   Val sequences:   {len(X_val)}")
        print(f"   Test sequences:  {len(X_test)}")
        if len(X_train) > 0:
            print(f"   X shape: {X_train.shape}")
            print(f"   y shape: {y_train.shape} - [views, likes, shares, comments]")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }


def main():
    """Main entry point."""
    
    # Configuration theo yêu cầu thầy
    VIDEO_DATA_DIR = "video_data/list32/90"
    OUTPUT_DIR = "data_processing/processed"
    MAX_DAYS = 7  # 7 days = 168 hours
    
    # Initialize preparator
    # Model sẽ tự động dùng:
    # - 3 snapshots (giờ 1, 2, 3)
    # - 7 features: views, likes, shares, comments, hour_of_day, day_of_week, time_horizon
    # - Time horizons: 1-7 days (24, 48, 72, 96, 120, 144, 168 hours)
    preparator = LSTMDataPreparator(
        video_data_dir=VIDEO_DATA_DIR,
        output_dir=OUTPUT_DIR,
        max_days=MAX_DAYS
    )
    
    # Run pipeline
    sequences = preparator.run_pipeline()
    
    print("\n✨ Data is ready for LSTM training!")
    print(f"\n📖 Model Specification (theo yêu cầu thầy):")
    print(f"")
    print(f"  Input X:")
    print(f"    Shape: (n_samples, 3, 7)")
    print(f"    - 3 snapshots: giờ 1, 2, 3 sau khi đăng")
    print(f"    - 7 features: views, likes, shares, comments, hour_of_day, day_of_week, time_horizon")
    print(f"")
    print(f"  Output y:")
    print(f"    Shape: (n_samples, 4)")
    print(f"    - 4 metrics: views, likes, shares, comments")
    print(f"    - Tại time_horizon (1-7 days sau khi đăng)")
    print(f"")
    print(f"  Time horizons: 24h, 48h, 72h, 96h, 120h, 144h, 168h")
    print(f"                 (1 day, 2 days, 3 days, ..., 7 days)")
    print(f"")
    print(f"To load data:")
    print(f"  X_train = np.load('{OUTPUT_DIR}/X_train.npy')")
    print(f"  y_train = np.load('{OUTPUT_DIR}/y_train.npy')")
    print(f"  X_val = np.load('{OUTPUT_DIR}/X_val.npy')")
    print(f"  y_val = np.load('{OUTPUT_DIR}/y_val.npy')")


if __name__ == "__main__":
    main()
