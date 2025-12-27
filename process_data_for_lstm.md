Here’s a **looser, more forgiving version** that still keeps the dataset usable.

---

## Dataset requirements (relaxed)

**1️⃣ Time resolution**

* Try to keep **~30-minute steps**,
  but **anywhere between 15–90 minutes is fine**.

**2️⃣ Time alignment**

* Convert timestamps to **“hours since posted”**.
* No need to perfectly align videos to each other.

**3️⃣ Minimum data per video**

* Keep videos that have **at least a few hours** of data (≥ 6 points).
  Prefer more, but **don’t discard short ones**.

**4️⃣ Missing / irregular data**

* Missing points are OK.
* Interpolation is optional — only fill **big obvious gaps** if it helps.
* Do **not drop** videos just because they’re messy.

**5️⃣ Features per snapshot**

* Must have: **views**.
* Nice-to-have (if available): likes, comments, shares, follower count, posting time.

**6️⃣ Coverage window**

* Use any snapshots that occur **within the first 7 days**.
* They **do NOT** need to start at posting time.

**7️⃣ Sequence building**

* Build sliding windows from whatever history exists.
* Input length: **as much as is available** (even short sequences).
* Predict the next few steps (configurable).

**8️⃣ Targets**

* Ensure labels use **future values only** (no leakage).
* Cumulative or deltas — either is fine, just be consistent.

**9️⃣ Data cleaning**

* Remove duplicate timestamps within a video.
* Remove clearly broken values (e.g., negative views).

**🔟 Splits**

* Split **by video** (not timestamp).