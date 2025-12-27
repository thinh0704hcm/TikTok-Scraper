Got it — here’s a **concise, hand-off checklist** for preparing training data for the Keras LSTM.

---

## 📌 Data Requirements — LSTM (View/Share Momentum)

### 1️⃣ Base table

Each row = one crawl snapshot.

Required columns:

* `video_id`
* `posted_at`
* `scraped_at`
* `views`
* `likes`
* `shares`
* `comments`

Derived:

* `t_since_post = scraped_at - posted_at` (in hours)

---

### 2️⃣ Time ordering & integrity

* Sort by `(video_id, scraped_at)`
* Remove duplicates
* Enforce monotonic `t_since_post`

---

### 3️⃣ Resampling

* Fixed interval (e.g., **30 minutes**)
* Forward-fill metrics for gaps
* Drop sequences shorter than the window length

---

### 4️⃣ Scaling

* Standardize numeric features (fit on **train only**)
* Persist scalers + column order

---

### 5️⃣ Windowing (supervised sequences)

* Sequence length: **L = 12** (example)
* Build sliding windows:

Shape:

* **X:** `(N, L, F)`
* **y:** aligned to target horizon (e.g., views at 24h / 7d)

---

### 6️⃣ Targets

Produce labels:

* `views_at_6h / 12h / 24h / 7d`
* optional: `viral_label` from Viral_Ratio rule

---

### 7️⃣ Splits

* Split **by time** (old → new)
* No leakage across videos

---

### 8️⃣ Save artifacts

* scalers
* feature list
* mapping from `video_id` → sequences
* train/val/test indices