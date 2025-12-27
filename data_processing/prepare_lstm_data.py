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
    Prepares TikTok video data for LSTM model training with RELAXED criteria.
    
    Key relaxations:
    - Accept any time resolution (15-90 min)
    - Minimum 6 snapshots per video (down from 12)
    - Use any data within first 7 days
    - No strict alignment required
    - Handle missing/irregular data gracefully
    """
    
    def __init__(self, 
                 video_data_dir: str,
                 output_dir: str = "data_processing/processed",
                 min_snapshots: int = 6,
                 sequence_length: int = 4,
                 max_days: int = 7):  # Use snapshots within first 7 days
        """
        Initialize the data preparator with relaxed criteria.
        
        Args:
            video_data_dir: Root directory containing video data
            output_dir: Directory to save processed data and artifacts
            min_snapshots: Minimum snapshots per video (relaxed to 6)
            sequence_length: Length of input sequences for LSTM
            max_days: Maximum days since posting to include (7 days = 168h)
        """
        self.video_data_dir = Path(video_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_snapshots = min_snapshots
        self.sequence_length = sequence_length
        self.max_days = max_days
        self.max_hours = max_days * 24
        
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
        
        print(f"✅ Data cleaned: {len(df)} records remaining")
        print(f"   Unique videos: {df['video_id'].nunique()}")
        print(f"   Unique accounts: {df['author_username'].nunique()}")
        
        return df
    
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
        
        return synthetic
    
    
    def filter_videos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter videos with relaxed criteria - keep videos with minimum snapshots.
        Accept any time resolution (15-90 min is fine, no strict requirement).
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Filtered DataFrame
        """
        print(f"\n⏱️  Filtering videos (relaxed: ≥{self.min_snapshots} snapshots, any within {self.max_days} days)...")
        
        filtered_groups = []
        
        for video_id, group in df.groupby('video_id'):
            group = group.sort_values('scraped_at')
            
            # Track account stats
            account = group['author_username'].iloc[0]
            self.account_stats[account]['videos'] += 1
            
            # Relaxed: just need minimum snapshots
            if len(group) < self.min_snapshots:
                continue
            
            filtered_groups.append(group)
        
        if not filtered_groups:
            raise ValueError(f"No videos have ≥{self.min_snapshots} snapshots!")
        
        df_filtered = pd.concat(filtered_groups, ignore_index=True)
        
        print(f"✅ Kept {len(filtered_groups)} videos with sufficient data")
        print(f"   Total records: {len(df_filtered)}")
        
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
        Create additional features for LSTM input.
        
        Args:
            df: Resampled DataFrame
            
        Returns:
            DataFrame with additional features
        """
        print("\n🔧 Creating features...")
        
        # Engagement rate features
        df['like_rate'] = df['likes'] / (df['views'] + 1)
        df['share_rate'] = df['shares'] / (df['views'] + 1)
        df['comment_rate'] = df['comments'] / (df['views'] + 1)
        df['collect_rate'] = df['collects'] / (df['views'] + 1)
        
        # Velocity features (per video)
        df = df.groupby('video_id').apply(self._compute_velocity).reset_index(drop=True)
        
        # Time-based features
        df['hour_of_day'] = df['scraped_at'].dt.hour
        df['day_of_week'] = df['scraped_at'].dt.dayofweek
        
        print(f"✅ Features created")
        
        return df
    
    def _compute_velocity(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute growth velocity features."""
        group = group.sort_values('scraped_at')
        
        # Time diff in hours
        time_diffs = group['scraped_at'].diff().dt.total_seconds() / 3600
        
        # Views velocity (views gained per hour)
        group['views_velocity'] = group['views'].diff() / time_diffs
        group['likes_velocity'] = group['likes'].diff() / time_diffs
        group['shares_velocity'] = group['shares'].diff() / time_diffs
        
        # Fill first row and infinities with 0
        group[['views_velocity', 'likes_velocity', 'shares_velocity']] = \
            group[['views_velocity', 'likes_velocity', 'shares_velocity']].fillna(0)
        group[['views_velocity', 'likes_velocity', 'shares_velocity']] = \
            group[['views_velocity', 'likes_velocity', 'shares_velocity']].replace([np.inf, -np.inf], 0)
        
        return group
    
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create next-step prediction targets (simple and always available).
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with target columns
        """
        print(f"\n🎯 Creating prediction targets (next-step views)...")
        
        df['target_views_next'] = np.nan
        df['target_views_change'] = np.nan
        
        for video_id, group in df.groupby('video_id'):
            group = group.sort_values('scraped_at')
            indices = group.index.tolist()
            
            for i in range(len(indices) - 1):
                current_idx = indices[i]
                next_idx = indices[i + 1]
                
                df.at[current_idx, 'target_views_next'] = df.at[next_idx, 'views']
                df.at[current_idx, 'target_views_change'] = df.at[next_idx, 'views'] - df.at[current_idx, 'views']
        
        # Create viral label based on view thresholds
        df['viral_label'] = (df['views'] > 500_000).astype(int)
        
        target_count = df['target_views_next'].notna().sum()
        print(f"✅ Created {target_count} target labels")
        
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
        Scale numeric features using StandardScaler (fit on train only).
        Keep t_since_post, scraped_at, posted_at, video_id, author_username unscaled.
        
        Args:
            train_df, val_df, test_df: Split DataFrames
            
        Returns:
            Tuple of scaled DataFrames
        """
        print("\n📊 Scaling features...")
        
        # Define features to scale (exclude t_since_post from scaling - needed for time filtering)
        self.feature_columns = [
            'views', 'likes', 'shares', 'comments', 'collects',
            't_since_post',  # Include in feature list for LSTM but don't scale
            'like_rate', 'share_rate', 'comment_rate', 'collect_rate',
            'views_velocity', 'likes_velocity', 'shares_velocity',
            'hour_of_day', 'day_of_week'
        ]
        
        # Features to actually scale (all except those needed for filtering/tracking)
        scale_columns = [
            'views', 'likes', 'shares', 'comments', 'collects',
            'like_rate', 'share_rate', 'comment_rate', 'collect_rate',
            'views_velocity', 'likes_velocity', 'shares_velocity',
            'hour_of_day', 'day_of_week'
        ]
        
        # Handle empty val/test sets
        if len(train_df) == 0:
            raise ValueError("Train set is empty!")
        
        # Fit scaler on training data only
        self.scaler.fit(train_df[scale_columns])
        
        # Transform all splits (but preserve unscaled t_since_post for filtering)
        train_df_scaled = train_df.copy()
        train_df_scaled[scale_columns] = self.scaler.transform(train_df[scale_columns])
        
        val_df_scaled = val_df.copy() if len(val_df) > 0 else pd.DataFrame()
        if len(val_df) > 0:
            val_df_scaled[scale_columns] = self.scaler.transform(val_df[scale_columns])
        
        test_df_scaled = test_df.copy() if len(test_df) > 0 else pd.DataFrame()
        if len(test_df) > 0:
            test_df_scaled[scale_columns] = self.scaler.transform(test_df[scale_columns])
        
        print(f"✅ Features scaled: {len(scale_columns)} features (t_since_post kept unscaled)")
        
        return train_df_scaled, val_df_scaled, test_df_scaled
    
    def create_sequences(self, df: pd.DataFrame, 
                         target_col: str = 'target_views_next') -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Create sliding window sequences from available history (relaxed approach).
        Build sequences from whatever history exists - even short ones.
        
        Args:
            df: Scaled DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (X, y, video_ids, account_names) where:
                X: shape (N, sequence_length, num_features)
                y: shape (N,)
                video_ids: list of video_ids for each sequence
                account_names: list of account names for each sequence
        """
        print(f"\n🪟 Creating sequences (length={self.sequence_length}, target={target_col})...")
        
        X_list = []
        y_list = []
        video_id_list = []
        account_list = []
        
        videos_with_sequences = set()
        
        for video_id, group in df.groupby('video_id'):
            group = group.sort_values('scraped_at')
            account = group['author_username'].iloc[0]
            
            # Drop rows with missing target
            group_valid = group.dropna(subset=[target_col])
            
            if len(group_valid) < self.sequence_length:
                continue
            
            # Extract feature matrix
            features = group_valid[self.feature_columns].values
            targets = group_valid[target_col].values
            
            # Create sliding windows - as many as available
            num_sequences = 0
            for i in range(len(group_valid) - self.sequence_length + 1):
                X_window = features[i:i + self.sequence_length]
                y_target = targets[i + self.sequence_length - 1]
                
                X_list.append(X_window)
                y_list.append(y_target)
                video_id_list.append(video_id)
                account_list.append(account)
                num_sequences += 1
            
            if num_sequences > 0:
                videos_with_sequences.add(video_id)
                self.account_stats[account]['sequences'] += num_sequences
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"✅ Created {len(X)} sequences from {len(videos_with_sequences)} videos")
        if len(X) > 0:
            print(f"   X shape: {X.shape}")
            print(f"   y shape: {y.shape}")
        
        return X, y, video_id_list, account_list
    
    def save_artifacts(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                       train_accounts: List[str], val_accounts: List[str], test_accounts: List[str]):
        """
        Save all artifacts for later use, including account statistics.
        
        Args:
            train_df, val_df, test_df: Processed DataFrames
            train_accounts, val_accounts, test_accounts: Account lists per split
        """
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
            'min_snapshots': self.min_snapshots,
            'max_days': self.max_days,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
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
        report_lines.append(f"  - Min snapshots per video: {self.min_snapshots}")
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
    
    def run_pipeline(self, target_col: str = 'target_views_next'):
        """
        Run the complete data preparation pipeline with relaxed criteria.
        
        Args:
            target_col: Target column for sequence creation
            
        Returns:
            Dictionary with train/val/test sequences
        """
        print("=" * 60)
        print("🚀 Starting LSTM Data Preparation Pipeline (RELAXED)")
        print("=" * 60)
        
        # 1. Load data
        self.raw_data = self.load_video_data()
        
        # 2. Clean and prepare
        df = self.clean_and_prepare(self.raw_data)
        
        # 3. Augment data for accounts with insufficient videos (MVP approach)
        # Each account gets 30 + random(0-10) videos for natural variety
        df = self.augment_videos_for_accounts(df, min_videos_base=30)
        
        # 4. Filter videos (relaxed criteria)
        df = self.filter_videos(df)
        
        # 5. Create features
        df = self.create_features(df)
        
        # 6. Create targets
        df = self.create_targets(df)
        
        # 7. Split data by video
        train_df, val_df, test_df = self.split_data(df)
        
        # 8. Scale features
        train_df_scaled, val_df_scaled, test_df_scaled = self.scale_features(
            train_df, val_df, test_df
        )
        
        # 9. Create sequences
        X_train, y_train, train_video_ids, train_accounts = self.create_sequences(train_df_scaled, target_col)
        X_val, y_val, val_video_ids, val_accounts = self.create_sequences(val_df_scaled, target_col)
        X_test, y_test, test_video_ids, test_accounts = self.create_sequences(test_df_scaled, target_col)
        
        # 10. Save artifacts (including account stats)
        self.save_artifacts(train_df_scaled, val_df_scaled, test_df_scaled,
                           train_accounts, val_accounts, test_accounts)
        
        # Save sequences
        np.save(self.output_dir / 'X_train.npy', X_train)
        np.save(self.output_dir / 'y_train.npy', y_train)
        np.save(self.output_dir / 'X_val.npy', X_val)
        np.save(self.output_dir / 'y_val.npy', y_val)
        np.save(self.output_dir / 'X_test.npy', X_test)
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
            print(f"   Sequence shape:  {X_train.shape[1:]} (timesteps, features)")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }


def main():
    """Main entry point."""
    
    # Configuration for RELAXED data processing
    VIDEO_DATA_DIR = "video_data/list32/90"  # Videos from 32 KOLs, posted in last 90 days
    OUTPUT_DIR = "data_processing/processed"
    MIN_SNAPSHOTS = 6  # Relaxed: minimum 6 snapshots per video (down from 12)
    SEQUENCE_LENGTH = 4  # Flexible: 4 timesteps for sequence
    MAX_DAYS = 7  # Use any snapshots within first 7 days
    TARGET_COL = "target_views_next"  # Predict next-step views
    
    # Initialize preparator
    preparator = LSTMDataPreparator(
        video_data_dir=VIDEO_DATA_DIR,
        output_dir=OUTPUT_DIR,
        min_snapshots=MIN_SNAPSHOTS,
        sequence_length=SEQUENCE_LENGTH,
        max_days=MAX_DAYS
    )
    
    # Run pipeline
    sequences = preparator.run_pipeline(target_col=TARGET_COL)
    
    print("\n✨ Data is ready for LSTM training!")
    print(f"\nTo load the data:")
    print(f"  X_train = np.load('{OUTPUT_DIR}/X_train.npy')")
    print(f"  y_train = np.load('{OUTPUT_DIR}/y_train.npy')")


if __name__ == "__main__":
    main()
