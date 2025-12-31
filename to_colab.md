# TikTok LSTM Model - Google Colab Guide

## Model Overview
**Input**: 3 snapshots (t=1h, 2h, 3h) × 7 features → **Predict** metrics at 1-7 days

- **Features** (7): views, likes, shares, comments, hour_of_day, day_of_week, target_horizon
- **Targets** (4): target_views, target_likes, target_shares, target_comments  
- **Horizons**: 24, 48, 72, 96, 120, 144, 168 hours (1-7 days)
- **Shape**: X=(N, 3, 7), y=(N, 4)

---

## 1. Prepare & Upload

```bash
cd data_processing
powershell Compress-Archive -Path processed -DestinationPath lstm_data.zip
```
Upload `lstm_data.zip` to Google Drive

---

## 2. Colab Setup

### Mount Drive & Extract
```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
zip_path = '/content/drive/MyDrive/lstm_data.zip'  # Update path
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall('/content/data')
```

### Load Data
```python
import numpy as np
import json

X_train = np.load('/content/data/processed/X_train.npy')
y_train = np.load('/content/data/processed/y_train.npy')
X_val = np.load('/content/data/processed/X_val.npy')
y_val = np.load('/content/data/processed/y_val.npy')
X_test = np.load('/content/data/processed/X_test.npy')
y_test = np.load('/content/data/processed/y_test.npy')

with open('/content/data/processed/metadata.json') as f:
    metadata = json.load(f)

print(f"X shape: {X_train.shape}")  # (N, 3, 7)
print(f"y shape: {y_train.shape}")  # (N, 4)
print(f"Features: {metadata['feature_columns']}")
```

---

## 3. Build LSTM Model

```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Model with named outputs
def build_kol_lstm(seq_length=3, n_features=7):
    inputs = layers.Input(shape=(seq_length, n_features))
    
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Named output heads
    out_views = layers.Dense(1, name='views')(x)
    out_likes = layers.Dense(1, name='likes')(x)
    out_shares = layers.Dense(1, name='shares')(x)
    out_comments = layers.Dense(1, name='comments')(x)
    
    model = models.Model(
        inputs=inputs,
        outputs=[out_views, out_likes, out_shares, out_comments]
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss={
            'views': 'mse',
            'likes': 'mse',
            'shares': 'mse',
            'comments': 'mse'
        },
        loss_weights={
            'views': 3.0,
            'likes': 1.0,
            'shares': 1.0,
            'comments': 1.0
        },
        metrics={
            'views': ['mae'],
            'likes': ['mae'],
            'shares': ['mae'],
            'comments': ['mae']
        }
    )
    
    return model

model = build_kol_lstm()
model.summary()
```

---

## 4. Train Per-KOL Models (Optional)

```python
import os
os.makedirs('kol_models', exist_ok=True)

# Load sequence info to filter by account
with open('/content/data/processed/sequence_info.json') as f:
    seq_info = json.load(f)

train_accounts = seq_info['train']['accounts']
val_accounts = seq_info['val']['accounts']
test_accounts = seq_info['test']['accounts']

# Get unique accounts
accounts = sorted(set(train_accounts))
print(f"Training {len(accounts)} KOL models")

results = []

for i, account in enumerate(accounts, 1):
    print(f"\n{'='*60}")
    print(f"🎯 [{i}/{len(accounts)}] Training: {account}")
    print(f"{'='*60}")
    
    # Filter data for this account
    train_mask = np.array(train_accounts) == account
    val_mask = np.array(val_accounts) == account
    test_mask = np.array(test_accounts) == account
    
    X_train_kol = X_train[train_mask]
    y_train_kol = y_train[train_mask]
    X_val_kol = X_val[val_mask]
    y_val_kol = y_val[val_mask]
    X_test_kol = X_test[test_mask]
    y_test_kol = y_test[test_mask]
    
    print(f"   Train: {len(X_train_kol)} | Val: {len(X_val_kol)} | Test: {len(X_test_kol)}")
    
    if len(X_train_kol) < 10:
        print(f"   ⚠️ Skipping {account} (insufficient data)")
        continue
    
    # Build model
    model = build_kol_lstm()
    
    # Callbacks
    checkpoint_path = f'kol_models/{account}_best.keras'
    checkpoint = callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0
    )
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
    )
    
    # Prepare y data as dict
    y_train_dict = {
        'views': y_train_kol[:, 0],
        'likes': y_train_kol[:, 1],
        'shares': y_train_kol[:, 2],
        'comments': y_train_kol[:, 3]
    }
    y_val_dict = {
        'views': y_val_kol[:, 0],
        'likes': y_val_kol[:, 1],
        'shares': y_val_kol[:, 2],
        'comments': y_val_kol[:, 3]
    }
    
    # Train
    history = model.fit(
        X_train_kol, y_train_dict,
        validation_data=(X_val_kol, y_val_dict),
        epochs=100,
        batch_size=16,
        callbacks=[checkpoint, early_stop],
        verbose=0
    )
    
    # Load best model
    if os.path.exists(checkpoint_path):
        model = keras.models.load_model(checkpoint_path)
    
    # Predict
    predictions = model.predict(X_test_kol, verbose=0)
    y_pred = np.column_stack(predictions)
    
    # Evaluate
    try:
        mae = mean_absolute_error(y_test_kol, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_kol, y_pred))
        r2 = r2_score(y_test_kol, y_pred)
        
        print(f"   ✅ Overall - MAE: {mae:,.0f} | RMSE: {rmse:,.0f} | R²: {r2:.4f}")
        
        # Per-target metrics
        names = ['views', 'likes', 'shares', 'comments']
        for j, name in enumerate(names):
            mae_j = mean_absolute_error(y_test_kol[:, j], y_pred[:, j])
            r2_j = r2_score(y_test_kol[:, j], y_pred[:, j])
            print(f"      {name.capitalize():10} MAE={mae_j:,.0f}  R²={r2_j:.3f}")
        
        results.append({
            'account': account,
            'test_mae': mae,
            'test_rmse': rmse,
            'test_r2': r2
        })
    
    except Exception as e:
        print(f"   ❌ Error: {e}")

# Summary
import pandas as pd
if results:
    results_df = pd.DataFrame(results).sort_values('test_mae')
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY - PER-KOL MODEL RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    print(f"\n📈 Average MAE: {results_df['test_mae'].mean():,.0f}")
    print(f"📈 Average R²: {results_df['test_r2'].mean():.4f}")
else:
    print("\n⚠️ No models were successfully trained")
```

---

## 5. Train Global Model

```python
# Prepare y data as dict for named outputs
y_train_dict = {
    'views': y_train[:, 0],
    'likes': y_train[:, 1],
    'shares': y_train[:, 2],
    'comments': y_train[:, 3]
}

y_val_dict = {
    'views': y_val[:, 0],
    'likes': y_val[:, 1],
    'shares': y_val[:, 2],
    'comments': y_val[:, 3]
}

# Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# Train
history = model.fit(
    X_train, y_train_dict,
    validation_data=(X_val, y_val_dict),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['views_mae'], label='Views MAE')
plt.plot(history.history['likes_mae'], label='Likes MAE')
plt.plot(history.history['shares_mae'], label='Shares MAE')
plt.plot(history.history['comments_mae'], label='Comments MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()
```

---

## 6. Evaluate Global Model

```python
from sklearn.metrics import mean_absolute_error, r2_score

# Predict (returns list of arrays for each output)
predictions = model.predict(X_test)
y_pred = np.column_stack(predictions)  # Combine to (N, 4)

# Overall
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MAE: {mae:,.0f}")
print(f"Test R²: {r2:.4f}\n")

# Per metric
targets = ['Views', 'Likes', 'Shares', 'Comments']
for i, name in enumerate(targets):
    mae_i = mean_absolute_error(y_test[:, i], y_pred[:, i])
    r2_i = r2_score(y_test[:, i], y_pred[:, i])
    print(f"{name}: MAE={mae_i:,.0f}, R²={r2_i:.4f}")
```

---

## 7. Save Model

```python
model.save('/content/drive/MyDrive/tiktok_lstm_model.keras')
```

---

## Notes

- **Input**: 3 snapshots at t=1h, 2h, 3h × 7 features (includes `target_horizon`)
- **Output**: Metrics at specified horizon (1-7 days)
- **target_horizon** is scaled with other features → model learns time-dependent patterns
- Monotonic filtering ensures metrics increase with time
