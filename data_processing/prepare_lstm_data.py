"""
LSTM Data Preparation Pipeline for TikTok Video Analytics

This script processes raw TikTok video data into sequences suitable for LSTM training.
It follows the data requirements specified in process_data_for_lstm.md
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
import warnings
warnings.filterwarnings('ignore')


class LSTMDataPreparator:
    """
    Prepares TikTok video data for LSTM model training.
    
    Handles:
    - Data loading from multiple JSON files
    - Time-based feature engineering
    - Resampling to fixed intervals
    - Sequence windowing
    - Train/val/test splitting
    - Feature scaling
    """
    
    def __init__(self, 
                 video_data_dir: str,
                 output_dir: str = "data_processing/processed",
                 resample_interval: str = "30min",
                 sequence_length: int = 12,
                 target_horizons: List[int] = [6, 12, 24, 168]):  # hours: 6h, 12h, 24h, 7d
        """
        Initialize the data preparator.
        
        Args:
            video_data_dir: Root directory containing video data
            output_dir: Directory to save processed data and artifacts
            resample_interval: Resampling interval (e.g., '30min', '1H')
            sequence_length: Length of input sequences for LSTM
            target_horizons: List of time horizons (in hours) for prediction targets
        """
        self.video_data_dir = Path(video_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.resample_interval = resample_interval
        self.sequence_length = sequence_length
        self.target_horizons = sorted(target_horizons)
        
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.raw_data = None
        self.processed_data = None
        
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
        Clean data and compute derived features.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame with derived features
        """
        print("\n🧹 Cleaning and preparing data...")
        
        # Remove rows with missing critical fields
        df = df.dropna(subset=['video_id', 'posted_at', 'scraped_at'])
        
        # Calculate time since post (in hours)
        df['t_since_post'] = (df['scraped_at'] - df['posted_at']).dt.total_seconds() / 3600
        
        # Remove negative or zero time differences (data issues)
        df = df[df['t_since_post'] > 0]
        
        # Sort by video_id and scraped_at
        df = df.sort_values(['video_id', 'scraped_at'])
        
        # Remove duplicates (same video_id and scraped_at)
        df = df.drop_duplicates(subset=['video_id', 'scraped_at'], keep='first')
        
        # Ensure monotonic t_since_post per video
        df = df.groupby('video_id').apply(self._ensure_monotonic).reset_index(drop=True)
        
        print(f"✅ Data cleaned: {len(df)} records remaining")
        print(f"   Unique videos: {df['video_id'].nunique()}")
        
        return df
    
    def _ensure_monotonic(self, group: pd.DataFrame) -> pd.DataFrame:
        """Ensure t_since_post is monotonically increasing within a video group."""
        group = group.sort_values('t_since_post')
        # Remove any rows where t_since_post decreases
        group = group[group['t_since_post'] >= group['t_since_post'].cummax()]
        return group
    
    def resample_videos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter videos to keep only those tracked from early stages (≤8h) with 24h+ tracking.
        This enables predicting 24h performance from early signals.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Filtered/Resampled DataFrame
        """
        print(f"\n⏱️  Processing video timelines for early-stage prediction...")
        
        filtered_groups = []
        early_cutoff = 12  # Must have measurements within first 12h (relaxed from 8h)
        target_horizon = 20  # Must be tracked to at least 20h
        
        for video_id, group in df.groupby('video_id'):
            group = group.sort_values('scraped_at')
            
            # Check if tracked from early stage (≤8h after posting)
            min_hours = group['t_since_post'].min()
            max_hours = group['t_since_post'].max()
            
            if min_hours > early_cutoff:
                continue  # Video too old when scraping started
            
            # Check if tracked long enough for 24h prediction
            if max_hours < target_horizon:
                continue  # Not tracked long enough
            
            # Check if video has minimum required snapshots
            if len(group) < self.sequence_length:
                continue
            
            # Optional: resample to regular intervals if requested
            if self.resample_interval and self.resample_interval != 'none':
                group = group.set_index('scraped_at')
                group = group.resample(self.resample_interval).last()
                group = group.ffill()
                group = group.reset_index()
                group['video_id'] = video_id
                
                # Recalculate t_since_post after resampling
                group['t_since_post'] = (group['scraped_at'] - group['posted_at']).dt.total_seconds() / 3600
                
                if len(group) < self.sequence_length:
                    continue
            
            filtered_groups.append(group)
        
        if not filtered_groups:
            raise ValueError(f"No videos meet criteria! Need: tracked from ≤{early_cutoff}h to ≥{target_horizon}h with ≥{self.sequence_length} measurements")
        
        df_processed = pd.concat(filtered_groups, ignore_index=True)
        
        print(f"✅ Found {len(filtered_groups)} videos suitable for early-stage → 24h prediction")
        print(f"   Total records: {len(df_processed)}")
        
        # Show timeline statistics
        timeline_stats = df_processed.groupby('video_id')['t_since_post'].agg(['min', 'max', 'count'])
        timeline_stats['span_hours'] = timeline_stats['max'] - timeline_stats['min']
        print(f"   Start of tracking: {timeline_stats['min'].min():.1f}h to {timeline_stats['min'].max():.1f}h after posting")
        print(f"   End of tracking: {timeline_stats['max'].min():.1f}h to {timeline_stats['max'].max():.1f}h after posting")
        print(f"   Snapshots per video: {timeline_stats['count'].min()}-{timeline_stats['count'].max()} (median: {timeline_stats['count'].median():.0f})")
        
        return df_processed
    
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
        
        # Views velocity (views gained per hour)
        group['views_velocity'] = group['views'].diff() / group['t_since_post'].diff()
        group['likes_velocity'] = group['likes'].diff() / group['t_since_post'].diff()
        group['shares_velocity'] = group['shares'].diff() / group['t_since_post'].diff()
        
        # Fill first row with 0
        group[['views_velocity', 'likes_velocity', 'shares_velocity']] = \
            group[['views_velocity', 'likes_velocity', 'shares_velocity']].fillna(0)
        
        return group
    
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target labels for different time horizons.
        Automatically detects data span and creates appropriate targets.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with target columns
        """
        print(f"\n🎯 Creating targets...")
        
        # Check data span per video
        time_spans = df.groupby('video_id')['t_since_post'].agg(['min', 'max'])
        time_spans['span'] = time_spans['max'] - time_spans['min']
        median_span = time_spans['span'].median()
        max_span = time_spans['span'].max()
        
        print(f"   Time span per video: median={median_span:.1f}h, max={max_span:.1f}h")
        
        # Filter horizons that are achievable with current data
        achievable_horizons = [h for h in self.target_horizons if h <= max_span]
        
        if not achievable_horizons:
            print(f"   ⚠️  WARNING: Data span ({max_span:.1f}h) is too short for any target horizons!")
            print(f"   Creating short-term prediction targets instead...")
            # Create next-step prediction (views at next timestep)
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
            
            # Create viral label based on current views
            df['viral_label'] = (df['views'] > 1_000_000).astype(int)
            
            # Count non-null targets
            target_count = df['target_views_next'].notna().sum()
            print(f"   ✅ Created next-step prediction targets: {target_count} samples")
            
            return df
        
        print(f"   Using horizons: {achievable_horizons} hours (others skipped due to data span)")
        
        # Create targets for achievable horizons
        targets_created = {}
        for horizon in achievable_horizons:
            target_col = f'views_at_{horizon}h'
            df[target_col] = np.nan
            count = 0
            
            for video_id, group in df.groupby('video_id'):
                group = group.sort_values('scraped_at')
                
                for idx, row in group.iterrows():
                    target_time = row['t_since_post'] + horizon
                    
                    # Find the closest measurement at or after target_time
                    future_rows = group[group['t_since_post'] >= target_time]
                    
                    if not future_rows.empty:
                        target_views = future_rows.iloc[0]['views']
                        df.at[idx, target_col] = target_views
                        count += 1
            
            targets_created[horizon] = count
        
        # Also create next-step prediction for short-term forecasting
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
        
        # Create viral label based on available data
        if achievable_horizons and max(achievable_horizons) >= 168:
            df['viral_label'] = (df['views_at_168h'] > 1_000_000).astype(int)
        elif achievable_horizons and max(achievable_horizons) >= 24:
            df['viral_label'] = (df['views_at_24h'] > 500_000).astype(int)
        else:
            df['viral_label'] = (df['views'] > 1_000_000).astype(int)
        
        print(f"   ✅ Targets created:")
        for horizon, count in targets_created.items():
            print(f"      views_at_{horizon}h: {count} samples")
        next_count = df['target_views_next'].notna().sum()
        print(f"      target_views_next: {next_count} samples")
        
        return df
    
    def split_data(self, df: pd.DataFrame, 
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by video_id to ensure each video's complete trajectory stays in one split.
        This prevents data leakage and ensures targets can be calculated properly.
        
        Args:
            df: Full DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"\n✂️  Splitting data by videos (train: {train_ratio}, val: {val_ratio}, test: {1-train_ratio-val_ratio})...")
        
        # Get unique video IDs and shuffle them for random split
        video_ids = df['video_id'].unique()
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(video_ids)
        
        # Split video IDs
        n_videos = len(video_ids)
        train_end = int(n_videos * train_ratio)
        val_end = int(n_videos * (train_ratio + val_ratio))
        
        train_videos = video_ids[:train_end]
        val_videos = video_ids[train_end:val_end]
        test_videos = video_ids[val_end:]
        
        # Split data based on video IDs
        train_df = df[df['video_id'].isin(train_videos)].copy()
        val_df = df[df['video_id'].isin(val_videos)].copy()
        test_df = df[df['video_id'].isin(test_videos)].copy()
        
        print(f"✅ Train: {len(train_df)} records ({len(train_videos)} videos)")
        print(f"   Val:   {len(val_df)} records ({len(val_videos)} videos)")
        print(f"   Test:  {len(test_df)} records ({len(test_videos)} videos)")
        
        return train_df, val_df, test_df
    
    def scale_features(self, train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale numeric features using StandardScaler (fit on train only).
        Keep t_since_post, scraped_at, posted_at, video_id unscaled for filtering.
        
        Args:
            train_df, val_df, test_df: Split DataFrames
            
        Returns:
            Tuple of scaled DataFrames
        """
        print("\n📊 Scaling features...")
        
        # Define features to scale (exclude t_since_post from scaling - needed for time filtering)
        self.feature_columns = [
            'views', 'likes', 'shares', 'comments', 'collects',
            't_since_post',  # Include in feature list for LSTM
            'like_rate', 'share_rate', 'comment_rate', 'collect_rate',
            'views_velocity', 'likes_velocity', 'shares_velocity',
            'hour_of_day', 'day_of_week'
        ]
        
        # Features to actually scale (all except those needed for filtering)
        scale_columns = [
            'views', 'likes', 'shares', 'comments', 'collects',
            'like_rate', 'share_rate', 'comment_rate', 'collect_rate',
            'views_velocity', 'likes_velocity', 'shares_velocity',
            'hour_of_day', 'day_of_week'
        ]
        
        # Fit scaler on training data only
        self.scaler.fit(train_df[scale_columns])
        
        # Transform all splits (but preserve unscaled t_since_post for filtering)
        train_df_scaled = train_df.copy()
        val_df_scaled = val_df.copy()
        test_df_scaled = test_df.copy()
        
        train_df_scaled[scale_columns] = self.scaler.transform(train_df[scale_columns])
        val_df_scaled[scale_columns] = self.scaler.transform(val_df[scale_columns])
        test_df_scaled[scale_columns] = self.scaler.transform(test_df[scale_columns])
        
        # Note: t_since_post remains unscaled for time-based filtering in create_sequences
        
        print(f"✅ Features scaled: {len(scale_columns)} features (t_since_post kept unscaled for filtering)")
        
        return train_df_scaled, val_df_scaled, test_df_scaled
    
    def create_sequences(self, df: pd.DataFrame, 
                         target_col: str = 'views_at_24h',
                         early_stage_cutoff: float = 12.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create sequences from EARLY measurements (≤12h) to predict 24h performance.
        Uses only early-stage data as input for realistic prediction scenario.
        
        Args:
            df: Scaled DataFrame
            target_col: Target column name
            early_stage_cutoff: Maximum hours since post for input sequences (default: 12h)
            
        Returns:
            Tuple of (X, y, video_ids) where:
                X: shape (N, sequence_length, num_features)
                y: shape (N,)
                video_ids: list of video_ids for each sequence
        """
        print(f"\n🪟 Creating early-stage sequences (≤{early_stage_cutoff}h) → predicting 24h views...")
        
        X_list = []
        y_list = []
        video_id_list = []
        
        skipped_reasons = {'no_early': 0, 'no_future': 0, 'success': 0}
        
        for video_id, group in df.groupby('video_id'):
            group = group.sort_values('scraped_at')
            
            # Split into early measurements (input) and target
            # Use UNSCALED t_since_post for time filtering
            early_group = group[group['t_since_post'] <= early_stage_cutoff].copy()
            
            # Need enough early measurements for a sequence
            if len(early_group) < self.sequence_length:
                skipped_reasons['no_early'] += 1
                continue
            
            # Get the target value (views at ~24h - find closest measurement)
            future_rows = group[group['t_since_post'] >= 20].copy()  # Relaxed from 24h
            if future_rows.empty:
                skipped_reasons['no_future'] += 1
                continue
            
            # Find measurement closest to 24h
            future_rows['distance_to_24h'] = abs(future_rows['t_since_post'] - 24)
            target_value = future_rows.loc[future_rows['distance_to_24h'].idxmin(), 'views']
            
            # Extract feature matrix from early measurements only
            features = early_group[self.feature_columns].values
            
            # Create sliding windows from early stage
            for i in range(len(early_group) - self.sequence_length + 1):
                X_window = features[i:i + self.sequence_length]
                
                X_list.append(X_window)
                y_list.append(target_value)
                video_id_list.append(video_id)
                skipped_reasons['success'] += 1
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"✅ Created {len(X)} sequences from {df['video_id'].nunique()} videos")
        if len(X) > 0:
            print(f"   X shape: {X.shape}")
            print(f"   y shape: {y.shape}")
        print(f"   Debug: no_early={skipped_reasons['no_early']}, no_future={skipped_reasons['no_future']}, success={skipped_reasons['success']}")
        
        return X, y, video_id_list
    
    def save_artifacts(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Save all artifacts for later use.
        
        Args:
            train_df, val_df, test_df: Processed DataFrames
        """
        print("\n💾 Saving artifacts...")
        
        # Save DataFrames
        train_df.to_csv(self.output_dir / 'train_data.csv', index=False)
        val_df.to_csv(self.output_dir / 'val_data.csv', index=False)
        test_df.to_csv(self.output_dir / 'test_data.csv', index=False)
        
        # Save scaler
        with open(self.output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'target_horizons': self.target_horizons,
            'resample_interval': self.resample_interval,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Artifacts saved to {self.output_dir}")
    
    def run_pipeline(self, target_col: str = 'views_at_24h'):
        """
        Run the complete data preparation pipeline.
        
        Args:
            target_col: Target column for sequence creation
            
        Returns:
            Dictionary with train/val/test sequences
        """
        print("=" * 60)
        print("🚀 Starting LSTM Data Preparation Pipeline")
        print("=" * 60)
        
        # 1. Load data
        self.raw_data = self.load_video_data()
        
        # 2. Clean and prepare
        df = self.clean_and_prepare(self.raw_data)
        
        # 3. Resample
        df = self.resample_videos(df)
        
        # 4. Create features
        df = self.create_features(df)
        
        # 5. Create targets
        df = self.create_targets(df)
        
        # 6. Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # 7. Scale features
        train_df_scaled, val_df_scaled, test_df_scaled = self.scale_features(
            train_df, val_df, test_df
        )
        
        # 8. Save artifacts
        self.save_artifacts(train_df_scaled, val_df_scaled, test_df_scaled)
        
        # 9. Create sequences
        X_train, y_train, train_video_ids = self.create_sequences(train_df_scaled, target_col)
        X_val, y_val, val_video_ids = self.create_sequences(val_df_scaled, target_col)
        X_test, y_test, test_video_ids = self.create_sequences(test_df_scaled, target_col)
        
        # Save sequences
        np.save(self.output_dir / 'X_train.npy', X_train)
        np.save(self.output_dir / 'y_train.npy', y_train)
        np.save(self.output_dir / 'X_val.npy', X_val)
        np.save(self.output_dir / 'y_val.npy', y_val)
        np.save(self.output_dir / 'X_test.npy', X_test)
        np.save(self.output_dir / 'y_test.npy', y_test)
        
        # Save video IDs
        with open(self.output_dir / 'video_ids.json', 'w') as f:
            json.dump({
                'train': train_video_ids,
                'val': val_video_ids,
                'test': test_video_ids
            }, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✅ Pipeline Complete!")
        print("=" * 60)
        print(f"\n📁 Output directory: {self.output_dir.absolute()}")
        print(f"\n📊 Summary:")
        print(f"   Train sequences: {len(X_train)}")
        print(f"   Val sequences:   {len(X_val)}")
        print(f"   Test sequences:  {len(X_test)}")
        print(f"   Sequence shape:  {X_train.shape[1:]} (timesteps, features)")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }


def main():
    """Main entry point."""
    
    # Configuration for Early-Stage → 24h Prediction
    VIDEO_DATA_DIR = "video_data/list32/90"  # Videos from 32 KOLs, posted in last 90 days
    OUTPUT_DIR = "data_processing/processed"
    RESAMPLE_INTERVAL = "none"  # Use raw measurements (every 30-45min) for maximum data
    SEQUENCE_LENGTH = 4  # 4 time steps from early stage (~2-3h of data) - relaxed to get more sequences
    TARGET_HORIZONS = [24]  # Predict views at 24 hours after posting
    TARGET_COL = "views_at_24h"  # Main target: views at 24h based on early signals (0-12h)
    
    # Initialize preparator
    preparator = LSTMDataPreparator(
        video_data_dir=VIDEO_DATA_DIR,
        output_dir=OUTPUT_DIR,
        resample_interval=RESAMPLE_INTERVAL,
        sequence_length=SEQUENCE_LENGTH,
        target_horizons=TARGET_HORIZONS
    )
    
    # Run pipeline
    sequences = preparator.run_pipeline(target_col=TARGET_COL)
    
    print("\n✨ Data is ready for LSTM training!")
    print(f"\nTo load the data:")
    print(f"  X_train = np.load('{OUTPUT_DIR}/X_train.npy')")
    print(f"  y_train = np.load('{OUTPUT_DIR}/y_train.npy')")


if __name__ == "__main__":
    main()
