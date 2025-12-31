# TikTok LSTM Model - Input/Output Specification

## Model Architecture Overview

Model dự đoán engagement metrics của video TikTok tại các thời điểm trong tương lai (1-7 ngày) dựa trên 3 snapshots đầu tiên (giờ 1, 2, 3).

---

## INPUT

### Shape
```
X = (N, 3, 7)
```
- **N**: Số lượng samples
- **3**: 3 snapshots tại t=1h, 2h, 3h sau khi đăng video
- **7**: 7 features cho mỗi snapshot

### Features (7 features per snapshot)

| Feature | Type | Description |
|---------|------|-------------|
| `views` | float | Số lượt xem tại thời điểm snapshot |
| `likes` | float | Số lượt thích tại thời điểm snapshot |
| `shares` | float | Số lượt chia sẻ tại thời điểm snapshot |
| `comments` | float | Số bình luận tại thời điểm snapshot |
| `hour_of_day` | int | Giờ trong ngày video được đăng (0-23) |
| `day_of_week` | int | Ngày trong tuần (0=Monday, 6=Sunday) |
| `target_horizon` | int | **Thời điểm dự đoán (đơn vị: GIỜ)** - Values: 24, 48, 72, 96, 120, 144, 168 |

### Example Input
```python
# Một sample có 3 snapshots
X[0] = [
    # Snapshot tại t=1h
    [1000,  50,  10,  5,  14,  2,  24],  # views, likes, shares, comments, hour, day, horizon
    
    # Snapshot tại t=2h
    [2500, 120,  25, 12,  14,  2,  24],
    
    # Snapshot tại t=3h
    [4000, 200,  40, 18,  14,  2,  24]
]
```

### Notes on Input
- **Temporal features** (hour_of_day, day_of_week) giúp model học patterns theo thời gian
- **target_horizon** là feature quan trọng cho model biết cần dự đoán bao xa
- Tất cả features được **scaled** bằng StandardScaler trước khi đưa vào model
- 3 snapshots đầu tiên capture **early growth pattern** của video

---

## OUTPUT

### Shape
```
y = (N, 4)
```
- **N**: Số lượng samples (giống input)
- **4**: 4 target metrics

### Targets (4 metrics)

| Target | Type | Description |
|--------|------|-------------|
| `target_views` | float | Số lượt xem dự đoán tại target_horizon |
| `target_likes` | float | Số lượt thích dự đoán tại target_horizon |
| `target_shares` | float | Số lượt chia sẻ dự đoán tại target_horizon |
| `target_comments` | float | Số bình luận dự đoán tại target_horizon |

### Prediction Horizons
Model có thể dự đoán tại các thời điểm (đơn vị: **giờ**):
- **24h** (1 ngày)
- **48h** (2 ngày)
- **72h** (3 ngày)
- **96h** (4 ngày)
- **120h** (5 ngày)
- **144h** (6 ngày)
- **168h** (7 ngày)

**Lưu ý**: `target_horizon` trong input có giá trị là **số giờ** (24, 48, 72...), không phải số ngày.

### Example Output
```python
# Dự đoán metrics tại 24h (1 ngày sau khi đăng)
y[0] = [15000, 800, 150, 60]  # views, likes, shares, comments
```

### Notes on Output
- Output được **inverse transform** từ scaled values về actual numbers
- **Monotonic constraint**: Metrics tại horizons xa hơn phải >= metrics tại horizons gần hơn
- Model dự đoán **absolute values**, không phải growth rate

---

## Model Usage Example

### Training
```python
# Load data
X_train = np.load('X_train.npy')  # Shape: (N, 3, 7)
y_train = np.load('y_train.npy')  # Shape: (N, 4)

# Build model
model = build_kol_lstm(seq_length=3, n_features=7)

# Train
model.fit(X_train, {
    'views': y_train[:, 0],
    'likes': y_train[:, 1],
    'shares': y_train[:, 2],
    'comments': y_train[:, 3]
})
```

### Inference
```python
# Prepare input: 3 snapshots × 7 features
input_data = np.array([[
    [1000, 50, 10, 5, 14, 2, 24],  # t=1h, predict at 24h
    [2500, 120, 25, 12, 14, 2, 24],  # t=2h
    [4000, 200, 40, 18, 14, 2, 24]   # t=3h
]])

# Scale input
input_scaled = scaler.transform(input_data.reshape(-1, 7)).reshape(1, 3, 7)

# Predict
predictions = model.predict(input_scaled)  # Returns [views, likes, shares, comments]

# Inverse transform to get actual numbers
y_pred = scaler_y.inverse_transform(predictions)
print(f"Predicted at 24h: views={y_pred[0]}, likes={y_pred[1]}, shares={y_pred[2]}, comments={y_pred[3]}")
```

---

## Data Pipeline

```
Raw Video Data
    ↓
Clean & Filter
    ↓
Create Snapshots (t=1h, 2h, 3h)
    ↓
Augment (Synthetic Videos)
    ↓
Expand with Horizons (×7 horizons)
    ↓
Add Temporal Features
    ↓
Train/Val/Test Split
    ↓
Scale Features
    ↓
Create Sequences
    ↓
X (N, 3, 7) → LSTM Model → y (N, 4)
```

---

## Key Characteristics

✅ **Multi-horizon prediction**: Single model predicts multiple time horizons  
✅ **Early signal focus**: Uses only first 3 hours of data  
✅ **Temporal awareness**: Includes posting time features  
✅ **Monotonic guarantee**: Ensures realistic growth patterns  
✅ **Scalable**: Works with limited data via augmentation  
