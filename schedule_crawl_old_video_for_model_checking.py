import subprocess
import time
from datetime import datetime
import sys

INTERVAL_SECONDS = 18000

while True:
    print(f"\n{'='*23}")
    print(f"Starting old video crawler at {datetime.now()}")
    print(f"{'='*23}\n")

    subprocess.run([
        sys.executable,
        "crawl_video.py",
        "--list", "list17",
        "--lookback", "720",
        "--no-resume"
    ])

    print(f"\nNext run in {INTERVAL_SECONDS/3600} hour...")
    time.sleep(INTERVAL_SECONDS)