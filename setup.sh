#!/bin/bash

echo "========================================"
echo "TikTok Scraper Docker Setup"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Docker
echo -e "\n${YELLOW}Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found. Please install Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker found${NC}"

# Check Docker Compose
echo -e "\n${YELLOW}Checking Docker Compose...${NC}"
if ! docker compose version &> /dev/null; then
    echo -e "${RED}✗ Docker Compose not found. Please install Docker Compose first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose found (V2)${NC}"

# Create necessary directories
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p data/profiles logs progress_tracking crawl_account
echo -e "${GREEN}✓ Directories created${NC}"

# Check for required files
echo -e "\n${YELLOW}Checking required files...${NC}"

FILES_MISSING=0

if [ ! -f "schedule_run_crawler.py" ]; then
    echo -e "${RED}✗ schedule_run_crawler.py not found${NC}"
    FILES_MISSING=1
fi

if [ ! -f "run_crawler.py" ]; then
    echo -e "${RED}✗ run_crawler.py not found${NC}"
    FILES_MISSING=1
fi

if [ ! -d "TT_Content_Scraper/src/scraper_functions" ]; then
    echo -e "${RED}✗ TT_Content_Scraper/src/scraper_functions/ directory not found${NC}"
    FILES_MISSING=1
fi

if [ ! -f "TT_Content_Scraper/src/scraper_functions/playwright_scraper.py" ]; then
    echo -e "${RED}✗ playwright_scraper.py not found${NC}"
    FILES_MISSING=1
fi

if [ ! -f "TT_Content_Scraper/src/object_tracker_db.py" ]; then
    echo -e "${RED}✗ object_tracker_db.py not found${NC}"
    FILES_MISSING=1
fi

if [ $FILES_MISSING -eq 1 ]; then
    echo -e "${RED}Please ensure all required files are present${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All required files present${NC}"

# Create sample account list if none exists
if [ ! -f "crawl_account/list32.txt" ]; then
    echo -e "\n${YELLOW}Creating sample account list...${NC}"
    cat > crawl_account/list32.txt << EOF
# Add TikTok usernames here (one per line)
# Example:
# tiktok
# khabylame
EOF
    echo -e "${GREEN}✓ Sample list created at crawl_account/list32.txt${NC}"
    echo -e "${YELLOW}⚠ Please edit crawl_account/list32.txt with your target accounts${NC}"
fi

# Create sample milestone file if none exists
if [ ! -f "milestone_datetime.txt" ]; then
    echo -e "\n${YELLOW}Creating milestone file...${NC}"
    # 90 days ago
    date -d "90 days ago" "+%Y-%m-%d 00:00:00" > milestone_datetime.txt 2>/dev/null || \
    date -v-90d "+%Y-%m-%d 00:00:00" > milestone_datetime.txt 2>/dev/null
    echo -e "${GREEN}✓ Milestone set to 90 days ago${NC}"
fi

# Check for optional files
echo -e "\n${YELLOW}Checking optional files...${NC}"

if [ ! -f "tiktok_cookies.json" ]; then
    echo -e "${YELLOW}⚠ tiktok_cookies.json not found (optional but recommended)${NC}"
    cat > tiktok_cookies.json << 'EOF'
[]
EOF
fi

if [ ! -f "browser_fingerprint.json" ]; then
    echo -e "${YELLOW}⚠ browser_fingerprint.json not found (using default)${NC}"
fi

# Set permissions
echo -e "\n${YELLOW}Setting permissions...${NC}"
chmod -R 755 data logs progress_tracking
echo -e "${GREEN}✓ Permissions set${NC}"

# Summary
echo -e "\n========================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "========================================\n"

echo "Next steps:"
echo "1. Edit crawl_account/list32.txt with your target accounts"
echo "2. (Optional) Add tiktok_cookies.json for authenticated scraping"
echo "3. Run: docker compose build"
echo "4. Run: docker compose up -d"
echo "5. Monitor: docker compose logs -f"
echo ""
echo "Or use Makefile commands:"
echo "  make build    - Build image"
echo "  make up       - Start container"
echo "  make logs     - View logs"
echo "  make data     - Check scraped data"
echo ""
echo -e "${GREEN}Happy scraping!${NC}"