import subprocess
import time
from datetime import datetime
import sys

# Run every 1 hour (3600 seconds)
INTERVAL_SECONDS = 3600

while True:
    print(f"\n{'='*20}")
    print(f"Starting video crawler at {datetime.now()}")
    print(f"{'='*20}\n")

    subprocess.run([
        sys.executable,
        "crawl_video.py",
        "--no-resume"
    ])

    print(f"\nNext run in {INTERVAL_SECONDS/3600} hour...")
    time.sleep(INTERVAL_SECONDS)