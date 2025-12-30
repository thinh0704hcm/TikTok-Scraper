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

# Load metadata
with open('/content/lstm_data/processed/metadata.json', 'r') as f:
    metadata = json.load(f)

train_meta = pd.read_csv('/content/lstm_data/processed/train_data.csv')
val_meta = pd.read_csv('/content/lstm_data/processed/val_data.csv')
test_meta = pd.read_csv('/content/lstm_data/processed/test_data.csv')

print(f"✅ Data loaded:")
print(f"   Train: {X_train.shape[0]:,} sequences")
print(f"   Val:   {X_val.shape[0]:,} sequences")
print(f"   Test:  {X_test.shape[0]:,} sequences")
print(f"   Accounts: {train_meta['author_username'].nunique()}")
print(f"   Features: {X_train.shape[2]} (includes target_horizon)")
print(f"   Horizons: {metadata['prediction_horizons']}")
```

---

## 5. Train Global Variable-Horizon Model

### Cell 7: Build Model
```python
def build_variable_horizon_lstm(seq_length=1, n_features=17):
    """
    LSTM that learns to predict at different horizons.
    The target_horizon feature tells the model how far ahead to predict.
    """
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(4)  # 4 outputs: [views, likes, shares, comments]
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

model = build_variable_horizon_lstm(
    seq_length=X_train.shape[1],
    n_features=X_train.shape[2]
)
model.summary()
```

### Cell 8: Train Model

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

# Model architecture
model = models.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(3, 7)),
    layers.Dropout(0.3),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(4)  # 4 targets
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()
```

---

## 4. Train Model

```python
# Callbacks
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Plot
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

---

## 5. Evaluate

```python
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)

# Overall
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MAE: {mae:,.0f}")
print(f"Test R²: {r2:.4f}")

# Per metric
targets = ['Views', 'Likes', 'Shares', 'Comments']
for i, name in enumerate(targets):
    mae_i = mean_absolute_error(y_test[:, i], y_pred[:, i])
    r2_i = r2_score(y_test[:, i], y_pred[:, i])
    print(f"{name}: MAE={mae_i:,.0f}, R²={r2_i:.4f}")
```

---

## 6. Save Model

```python
model.save('/content/drive/MyDrive/tiktok_lstm_model.keras')
```

---

## Notes

- **Input**: 3 snapshots at t=1h, 2h, 3h × 7 features (includes `target_horizon`)
- **Output**: Metrics at specified horizon (1-7 days)
- **target_horizon** is scaled with other features → model learns time-dependent patterns
- Monotonic filtering ensures metrics increase with time
