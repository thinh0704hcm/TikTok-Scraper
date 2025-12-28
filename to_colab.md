# Transfer Data to Google Colab

## 1. Prepare Data Locally

```bash
# Zip the processed data folder
cd data_processing
powershell Compress-Archive -Path processed -DestinationPath lstm_data.zip
```

## 2. Upload to Google Drive

1. Upload `lstm_data.zip` to your Google Drive (any folder)
2. Note the folder path (e.g., `/MyDrive/tiktok-lstm/`)

## 3. Setup Google Colab

Create new notebook, run these cells:

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Extract Data
```python
import zipfile
import os

# Update path to your zip location
zip_path = '/content/drive/MyDrive/tiktok-lstm/lstm_data.zip'
extract_to = '/content/lstm_data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"✅ Data extracted to: {extract_to}")
os.listdir(f"{extract_to}/processed")
```

### Cell 3: Verify Data
```python
import numpy as np

X_train = np.load('/content/lstm_data/processed/X_train.npy')
y_train = np.load('/content/lstm_data/processed/y_train.npy')

print(f"X_train shape: {X_train.shape}")  # (25132, 4, 15)
print(f"y_train shape: {y_train.shape}")  # (25132,)
```

---

# Build Keras LSTM Model

## Cell 4: Install Dependencies
```python
!pip install tensorflow scikit-learn matplotlib
```

## Cell 5: Import Libraries
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
```

## Cell 6: Load All Data
```python
# Load sequences
X_train = np.load('/content/lstm_data/processed/X_train.npy')
y_train = np.load('/content/lstm_data/processed/y_train.npy')
X_val = np.load('/content/lstm_data/processed/X_val.npy')
y_val = np.load('/content/lstm_data/processed/y_val.npy')
X_test = np.load('/content/lstm_data/processed/X_test.npy')
y_test = np.load('/content/lstm_data/processed/y_test.npy')

# Load metadata
train_meta = pd.read_csv('/content/lstm_data/processed/train_data.csv')
val_meta = pd.read_csv('/content/lstm_data/processed/val_data.csv')
test_meta = pd.read_csv('/content/lstm_data/processed/test_data.csv')

print(f"✅ Data loaded:")
print(f"   Train: {X_train.shape[0]} sequences")
print(f"   Val:   {X_val.shape[0]} sequences")
print(f"   Test:  {X_test.shape[0]} sequences")
print(f"   Accounts: {train_meta['author_username'].nunique()}")
```

## Cell 7: Build LSTM Model
```python
def build_lstm_model(seq_length=4, n_features=15):
    model = keras.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Regression output
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

model = build_lstm_model()
model.summary()
```

## Cell 8: Setup Callbacks
```python
checkpoint = callbacks.ModelCheckpoint(
    'best_lstm_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
```

## Cell 9: Train Model
```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)
```

## Cell 10: Plot Training History
```python
plt.figure(figsize=(14, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training vs Validation Loss')

# MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Training vs Validation MAE')

plt.tight_layout()
plt.show()
```

## Cell 11: Evaluate on Test Set
```python
# Load best model
model = keras.models.load_model('best_lstm_model.keras')

# Predict
y_pred = model.predict(X_test).flatten()

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("📊 Test Set Metrics:")
print(f"   MAE:  {mae:,.2f}")
print(f"   RMSE: {rmse:,.2f}")
print(f"   R²:   {r2:.4f}")
```

## Cell 12: Analyze Per-Account Performance
```python
# Add predictions to test metadata
test_meta['predictions'] = y_pred

# Per-account metrics
account_metrics = []
for account in test_meta['author_username'].unique():
    mask = test_meta['author_username'] == account
    y_true_acc = y_test[mask]
    y_pred_acc = y_pred[mask]
    
    account_metrics.append({
        'account': account,
        'n_sequences': mask.sum(),
        'mae': mean_absolute_error(y_true_acc, y_pred_acc),
        'rmse': np.sqrt(mean_squared_error(y_true_acc, y_pred_acc)),
        'r2': r2_score(y_true_acc, y_pred_acc)
    })

metrics_df = pd.DataFrame(account_metrics).sort_values('mae')
print(metrics_df.to_string(index=False))
```

## Cell 13: Save Results to Drive
```python
# Save model to Drive
model.save('/content/drive/MyDrive/tiktok-lstm/lstm_model_final.keras')

# Save metrics
metrics_df.to_csv('/content/drive/MyDrive/tiktok-lstm/per_account_metrics.csv', index=False)

# Save predictions
test_meta.to_csv('/content/drive/MyDrive/tiktok-lstm/test_predictions.csv', index=False)

print("✅ Saved to Google Drive!")
```

---

## Tips

- **Runtime**: Change to GPU (Runtime → Change runtime type → GPU → T4)
- **Training time**: ~10-20 min for 100 epochs with early stopping
- **Batch size**: Start with 32, increase to 64 if memory allows
- **Learning rate**: 0.001 is good start, scheduler will reduce if needed
- **Overfitting**: Add more dropout (0.3-0.4) if val_loss >> train_loss
