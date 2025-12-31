# Hướng dẫn chạy prepare_lstm_data.py

## Các thay đổi mới:

### 1. ✅ **Đã BỎ normalization/scaling features**
   - Features (views, likes, shares, comments, etc.) giờ sẽ giữ nguyên giá trị gốc
   - Không còn StandardScaler transform
   - Phù hợp cho một số mô hình không cần normalize

### 2. ✅ **Thêm tùy chọn TẮT data augmentation**
   - Mặc định: augmentation BẬT (tạo synthetic videos)
   - Có thể tắt để test nhanh với real data only

---

## Cách chạy:

### Chế độ NORMAL (có augmentation, không normalize):
```bash
python data_processing/prepare_lstm_data.py
```

### Chế độ TESTING NHANH (không augmentation, không normalize):
```bash
python data_processing/prepare_lstm_data.py --no-augmentation
```

---

## So sánh 2 chế độ:

| Tính năng | Normal | Testing (--no-augmentation) |
|-----------|--------|----------------------------|
| Normalization | ❌ TẮT | ❌ TẮT |
| Data Augmentation | ✅ BẬT | ❌ TẮT |
| Tốc độ | Chậm hơn | **Nhanh hơn** |
| Số lượng data | Nhiều hơn | Ít hơn (real only) |
| Phù hợp cho | Training production | Testing/debugging |

---

## Ví dụ output:

### Với augmentation:
```
🔬 Data augmentation: ENABLED
✅ Augmentation complete:
   Accounts augmented: 15/20
   Synthetic videos created: 250
   Total videos now: 500
```

### Không augmentation:
```
🔬 Data augmentation: DISABLED (using real data only)
```

---

## Lưu ý:

- **Normalization đã BỎ**: Data giữ nguyên giá trị gốc (views, likes, etc.)
- **--no-augmentation**: Dùng khi cần test nhanh, không cần synthetic data
- Sau khi process, data vẫn save vào `data_processing/processed/`
