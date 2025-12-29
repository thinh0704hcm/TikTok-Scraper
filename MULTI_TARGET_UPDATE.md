# Multi-Target Prediction Update

## Summary
Modified the data processing script to support **multi-target prediction** for views, likes, shares, and comments instead of only views.

## Changes Made

### 1. **create_targets()** function
- Added target columns for likes, shares, and comments
- Now creates 4 target columns:
  - `target_views_next`
  - `target_likes_next`
  - `target_shares_next`
  - `target_comments_next`

### 2. **create_sequences()** function
- Changed signature to accept `target_cols: List[str]` instead of single `target_col`
- Returns a dictionary of target arrays instead of a single array
- Output format: `(X, y_dict, video_id_list, account_list)` where `y_dict` contains all target arrays

### 3. **run_pipeline()** function
- Updated to accept `target_cols: List[str]` parameter
- Default: `['target_views_next', 'target_likes_next', 'target_shares_next', 'target_comments_next']`
- Saves separate `.npy` files for each target metric:
  - `y_train_target_views_next.npy`
  - `y_train_target_likes_next.npy`
  - `y_train_target_shares_next.npy`
  - `y_train_target_comments_next.npy`
  - (and corresponding `y_val_*` and `y_test_*` files)
- Also saves legacy `y_train.npy`, `y_val.npy`, `y_test.npy` for backward compatibility (contains views)

### 4. **save_artifacts()** function
- Updated metadata to include target column information
- Added `target_columns` and `target_description` fields

### 5. **main()** function
- Updated configuration to use `TARGET_COLS` list instead of single `TARGET_COL`
- Updated usage examples to show how to load each target

## Output Files

After running the script, you'll have:

### Feature Data
- `X_train.npy` - Training sequences
- `X_val.npy` - Validation sequences  
- `X_test.npy` - Test sequences

### Target Data (for each metric)
- `y_train_target_views_next.npy` / `y_val_target_views_next.npy` / `y_test_target_views_next.npy`
- `y_train_target_likes_next.npy` / `y_val_target_likes_next.npy` / `y_test_target_likes_next.npy`
- `y_train_target_shares_next.npy` / `y_val_target_shares_next.npy` / `y_test_target_shares_next.npy`
- `y_train_target_comments_next.npy` / `y_val_target_comments_next.npy` / `y_test_target_comments_next.npy`

### Legacy Files (backward compatibility)
- `y_train.npy` / `y_val.npy` / `y_test.npy` - Contains views data

### Metadata
- `metadata.json` - Updated with target column information

## Usage Example

```python
import numpy as np

# Load features
X_train = np.load('data_processing/processed/X_train.npy')
X_val = np.load('data_processing/processed/X_val.npy')
X_test = np.load('data_processing/processed/X_test.npy')

# Load targets for views prediction
y_train_views = np.load('data_processing/processed/y_train_target_views_next.npy')
y_val_views = np.load('data_processing/processed/y_val_target_views_next.npy')
y_test_views = np.load('data_processing/processed/y_test_target_views_next.npy')

# Load targets for likes prediction
y_train_likes = np.load('data_processing/processed/y_train_target_likes_next.npy')
y_val_likes = np.load('data_processing/processed/y_val_target_likes_next.npy')
y_test_likes = np.load('data_processing/processed/y_test_target_likes_next.npy')

# Load targets for shares prediction
y_train_shares = np.load('data_processing/processed/y_train_target_shares_next.npy')
y_val_shares = np.load('data_processing/processed/y_val_target_shares_next.npy')
y_test_shares = np.load('data_processing/processed/y_test_target_shares_next.npy')

# Load targets for comments prediction
y_train_comments = np.load('data_processing/processed/y_train_target_comments_next.npy')
y_val_comments = np.load('data_processing/processed/y_val_target_comments_next.npy')
y_test_comments = np.load('data_processing/processed/y_test_target_comments_next.npy')

# Train separate models or a multi-output model
# Example: Train model for views
model_views.fit(X_train, y_train_views, validation_data=(X_val, y_val_views))

# Example: Train model for likes  
model_likes.fit(X_train, y_train_likes, validation_data=(X_val, y_val_likes))
```

## Running the Script

To generate the new multi-target dataset:

```bash
python data_processing/prepare_lstm_data.py
```

The script will process all video data and create separate target files for each metric (views, likes, shares, comments).

## Backward Compatibility

The script maintains backward compatibility by:
1. Still saving `y_train.npy`, `y_val.npy`, `y_test.npy` files (containing views data)
2. Using the same feature columns as before
3. Maintaining the same data structure for X arrays

Existing code that only predicts views will continue to work without modification.
