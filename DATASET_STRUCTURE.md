# Dataset Structure

```
180/ # look_back_date
├── 20251223_234828/ # crawl_date_time
│   ├── run_summary_20251223_234828.json
│   ├── atusmain/
│   │   ├── summary_20251223_235134.json
│   │   └── videos_raw_20251223_235134.json
│   ├── bac_hello2/
│   │   ├── summary_YYYYMMDD_HHMMSS.json
│   │   └── videos_raw_YYYYMMDD_HHMMSS.json
│   ├── bnh.trng.lui/
│   │   ├── summary_YYYYMMDD_HHMMSS.json
│   │   └── videos_raw_YYYYMMDD_HHMMSS.json
│   ├── congb_/
│   │   ├── summary_YYYYMMDD_HHMMSS.json
│   │   └── videos_raw_YYYYMMDD_HHMMSS.json
│   ├── ...
│   └── qh.masterd/
│       ├── summary_YYYYMMDD_HHMMSS.json
│       └── videos_raw_YYYYMMDD_HHMMSS.json
├── 20251224_002637/
│   ├── run_summary_20251224_002637.json
│   ├── atusmain/
│   │   ├── summary_YYYYMMDD_HHMMSS.json
│   │   └── videos_raw_YYYYMMDD_HHMMSS.json
│   ├── bac_hello2/
│   │   ├── summary_YYYYMMDD_HHMMSS.json
│   │   └── videos_raw_YYYYMMDD_HHMMSS.json
│   └── ...
├── 20251224_013255/
│   ├── run_summary_20251224_013255.json
│   ├── atusmain/
│   │   ├── summary_YYYYMMDD_HHMMSS.json
│   │   └── videos_raw_YYYYMMDD_HHMMSS.json
│   └── ...
├── 20251224_033115/
│   ├── run_summary_20251224_033115.json
│   └── ...
├── 20251224_073514/
│   ├── run_summary_20251224_073514.json
│   └── ...
├── 20251224_090732/
│   ├── run_summary_20251224_090732.json
│   └── ...
├── 20251224_092721/
│   ├── run_summary_20251224_092721.json
│   └── ...
├── 20251224_094414/
│   ├── run_summary_20251224_094414.json
│   └── ...
├── 20251224_111050/
│   ├── run_summary_20251224_111050.json
│   └── ...
├── 20251224_111102/
│   ├── run_summary_20251224_111102.json
│   └── ...
└── 20251224_111115/
    ├── run_summary_20251224_111115.json
    └── ...
```

### run_summary_YYYYMMDD_HHMMSS.json
```json
{
  "run_name": "string",
  "lookback_days": 180,
  "output_dir": "string",
  "started_at": "timestamp",
  "profiles_processed": 0,
  "profiles_success": 0,
  "profiles_failed": 0,
  "total_videos": 0,
  "total_time_points": 0,
  "profile_summaries": {
    "username": {
      "username": "string",
      "videos": 0,
      "time_points": 0,
      "date_range": {
        "start": "date",
        "end": "date"
      },
      "files": {
        "videos": "string"
      }
    }
  }
}
```

## JSON Structure Example (atusmain/)

### summary_YYYYMMDD_HHMMSS.json
```json
{
  "username": "string",
  "lookback_days": 180,
  "date_range": {
    "start": "date",
    "end": "date"
  },
  "scraped_at": "timestamp",
  "total_videos": 0,
  "date_range_actual": {
    "earliest": "timestamp",
    "latest": "timestamp"
  },
  "total_stats": {
    "total_views": 0,
    "total_likes": 0,
    "total_comments": 0,
    "total_shares": 0
  },
  "average_stats": {
    "avg_views": 0.0,
    "avg_likes": 0.0,
    "avg_comments": 0.0
  },
  "files": {
    "videos": "string"
  },
  "scraper_stats": {
    "videos_scraped": 0,
    "profiles_scraped": 0,
    "errors": 0,
    "api_calls": 0
  }
}
```

### videos_raw_YYYYMMDD_HHMMSS.json
```json
[
  {
    "video_id": "string",
    "author_username": "string",
    "author_id": "string",
    "description": "string",
    "create_time": "timestamp",
    "create_timestamp": 0,
    "hashtags": ["string"],
    "stats": {
      "playCount": 0,
      "diggCount": 0,
      "commentCount": 0,
      "shareCount": 0,
      "collectCount": 0
    },
    "music_title": "string",
    "scraped_at": "timestamp"
  }
]
```
