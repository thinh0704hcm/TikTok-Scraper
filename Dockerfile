# Alternative Dockerfile using full Python image (larger but more compatible)
FROM python:3.11

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DISPLAY=:99 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    xvfb \
    x11-utils \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Optional: Install Brave Browser (uncomment to use Brave instead of Chromium)
RUN wget -q -O - https://brave-browser-apt-release.s3.brave.com/brave-browser-archive-keyring.gpg | gpg --dearmor -o /usr/share/keyrings/brave-browser-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/brave-browser-archive-keyring.gpg] https://brave-browser-apt-release.s3.brave.com/ stable main" | tee /etc/apt/sources.list.d/brave-browser-release.list && \
    apt-get update && \
    apt-get install -y brave-browser && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies and Chromium browser
# Note: We install system dependencies manually above, so we skip playwright install-deps
# to avoid font package conflicts in Debian slim
RUN pip install --no-cache-dir -r requirements.txt && \
    playwright install chromium

# Copy application files
COPY TT_Content_Scraper/ ./TT_Content_Scraper/
COPY schedule_run_crawler.py .
COPY run_crawler.py .

# Create necessary directories
RUN mkdir -p data/profiles logs progress_tracking crawl_account

# Create entrypoint script
RUN echo '#!/bin/bash\n\
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &\n\
sleep 2\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "schedule_run_crawler.py"]