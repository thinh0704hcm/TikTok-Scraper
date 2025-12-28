# Per-KOL LSTM Models on Google Colab

## Setup (Same as Global Model)

Follow Cells 1-6 from `to_colab.md` to:
1. Mount Drive and extract data
2. Load sequences (X_train, y_train, etc.)
3. Load metadata (train_data.csv, val_data.csv, test_data.csv)

---

## Cell 7: Build Per-KOL Model Function
```python
def build_kol_lstm(seq_length=4, n_features=15):
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model
```

## Cell 8: Get Unique Accounts
```python
accounts = sorted(train_meta['author_username'].unique())
print(f"📊 Training {len(accounts)} per-KOL models")
print(f"   Accounts: {accounts}")
```

## Cell 9: Train All Per-KOL Models
```python
import os
os.makedirs('kol_models', exist_ok=True)

results = []

for i, account in enumerate(accounts, 1):
    print(f"\n{'='*60}")
    print(f"🎯 [{i}/{len(accounts)}] Training: {account}")
    print(f"{'='*60}")
    
    # Filter data for this account
    train_mask = train_meta['author_username'] == account
    val_mask = val_meta['author_username'] == account
    test_mask = test_meta['author_username'] == account
    
    X_train_kol = X_train[train_mask]
    y_train_kol = y_train[train_mask]
    X_val_kol = X_val[val_mask]
    y_val_kol = y_val[val_mask]
    X_test_kol = X_test[test_mask]
    y_test_kol = y_test[test_mask]
    
    print(f"   Train: {len(X_train_kol)} | Val: {len(X_val_kol)} | Test: {len(X_test_kol)}")
    
    # Build model
    model = build_kol_lstm()
    
    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        f'kol_models/{account}_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train
    history = model.fit(
        X_train_kol, y_train_kol,
        validation_data=(X_val_kol, y_val_kol),
        epochs=100,
        batch_size=16,
        callbacks=[checkpoint, early_stop],
        verbose=0
    )
    
    # Evaluate
    model = keras.models.load_model(f'kol_models/{account}_best.keras')
    y_pred = model.predict(X_test_kol, verbose=0).flatten()
    
    mae = mean_absolute_error(y_test_kol, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_kol, y_pred))
    r2 = r2_score(y_test_kol, y_pred)
    
    results.append({
        'account': account,
        'train_size': len(X_train_kol),
        'val_size': len(X_val_kol),
        'test_size': len(X_test_kol),
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'test_mae': mae,
        'test_rmse': rmse,
        'test_r2': r2
    })
    
    print(f"   ✅ Test MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | R²: {r2:.4f}")

print(f"\n{'='*60}")
print(f"✅ All {len(accounts)} models trained!")
print(f"{'='*60}")
```

## Cell 10: View Results Summary
```python
results_df = pd.DataFrame(results)

print("\n📊 PER-KOL MODEL PERFORMANCE SUMMARY")
print("="*80)
print(results_df.sort_values('test_mae').to_string(index=False))

print("\n📈 AGGREGATE STATISTICS:")
print(f"   Mean MAE:  {results_df['test_mae'].mean():,.2f} ± {results_df['test_mae'].std():.2f}")
print(f"   Mean RMSE: {results_df['test_rmse'].mean():,.2f} ± {results_df['test_rmse'].std():.2f}")
print(f"   Mean R²:   {results_df['test_r2'].mean():.4f} ± {results_df['test_r2'].std():.4f}")
print(f"\n   Best performer:  {results_df.loc[results_df['test_mae'].idxmin(), 'account']} (MAE: {results_df['test_mae'].min():,.2f})")
print(f"   Worst performer: {results_df.loc[results_df['test_mae'].idxmax(), 'account']} (MAE: {results_df['test_mae'].max():,.2f})")
```

## Cell 11: Plot Per-KOL Performance
```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MAE
results_df.sort_values('test_mae').plot(
    x='account', y='test_mae', kind='barh', ax=axes[0], legend=False, color='steelblue'
)
axes[0].set_title('Mean Absolute Error (MAE)')
axes[0].set_xlabel('MAE')
axes[0].set_ylabel('')

# RMSE
results_df.sort_values('test_rmse').plot(
    x='account', y='test_rmse', kind='barh', ax=axes[1], legend=False, color='coral'
)
axes[1].set_title('Root Mean Squared Error (RMSE)')
axes[1].set_xlabel('RMSE')
axes[1].set_ylabel('')

# R²
results_df.sort_values('test_r2', ascending=False).plot(
    x='account', y='test_r2', kind='barh', ax=axes[2], legend=False, color='seagreen'
)
axes[2].set_title('R² Score')
axes[2].set_xlabel('R²')
axes[2].set_ylabel('')

plt.tight_layout()
plt.show()
```

## Cell 12: Make Predictions with Per-KOL Models
```python
# Load all models and predict
all_predictions = []

for account in accounts:
    # Load model
    model = keras.models.load_model(f'kol_models/{account}_best.keras')
    
    # Get test data for this account
    test_mask = test_meta['author_username'] == account
    X_test_kol = X_test[test_mask]
    
    # Predict
    preds = model.predict(X_test_kol, verbose=0).flatten()
    
    # Store with metadata
    test_meta_kol = test_meta[test_mask].copy()
    test_meta_kol['predictions'] = preds
    test_meta_kol['actual'] = y_test[test_mask]
    
    all_predictions.append(test_meta_kol)

# Combine all predictions
predictions_df = pd.concat(all_predictions, ignore_index=True)

print(f"✅ Generated {len(predictions_df)} predictions across {len(accounts)} accounts")
print(predictions_df.head(10))
```

## Cell 13: Save Models & Results to Drive
```python
import shutil

# Create Drive folder
drive_folder = '/content/drive/MyDrive/tiktok-lstm/per_kol_models'
os.makedirs(drive_folder, exist_ok=True)

# Copy all models
shutil.copytree('kol_models', f'{drive_folder}/models', dirs_exist_ok=True)

# Save results
results_df.to_csv(f'{drive_folder}/kol_performance.csv', index=False)
predictions_df.to_csv(f'{drive_folder}/all_predictions.csv', index=False)

print(f"✅ Saved to: {drive_folder}")
print(f"   • {len(accounts)} model files (.keras)")
print(f"   • kol_performance.csv")
print(f"   • all_predictions.csv")
```

---

## Comparison: Global vs Per-KOL

| Aspect | Global Model | Per-KOL Models |
|--------|-------------|----------------|
| Training time | ~15-20 min | ~30-60 min (30 models) |
| Model size | 1 model (~5MB) | 30 models (~150MB total) |
| Personalization | Low | High |
| Data efficiency | High | Moderate (each model sees less data) |
| Best for | General patterns | Account-specific behavior |

**Use Per-KOL when:**
- Accounts have distinct viral patterns
- You need personalized predictions
- You have sufficient data per account (30+ videos)

**Use Global when:**
- Limited data per account
- Want faster training
- Accounts share similar growth patterns
