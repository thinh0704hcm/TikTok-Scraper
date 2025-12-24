import subprocess
import time
from datetime import datetime
import sys

INTERVAL_SECONDS = 3600  # 1 hour

while True:
    print(f"\n{'='*60}")
    print(f"Starting crawler at {datetime.now()}")
    print(f"{'='*60}\n")

    subprocess.run([
        sys.executable,      # <-- venv python
        "run_crawler.py",
        "--no-resume"
    ])

    print("\nNext run in 1 hour...")
    time.sleep(INTERVAL_SECONDS)
