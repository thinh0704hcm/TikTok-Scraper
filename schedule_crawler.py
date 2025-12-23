"""Run crawler every 1 hour"""
import subprocess
import time
from datetime import datetime

INTERVAL_SECONDS = 3600  # 1 hour

while True:
    print(f"\n{'='*60}")
    print(f"Starting crawler at {datetime.now()}")
    print(f"{'='*60}\n")
    
    subprocess.run(["python", "run_crawler.py"])
    
    print(f"\nNext run in 1 hour...")
    time.sleep(INTERVAL_SECONDS)
