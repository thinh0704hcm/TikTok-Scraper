.PHONY: help build up down restart logs shell data stats clean backup

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	docker compose build

up: ## Start container in background
	docker compose up -d

down: ## Stop and remove container
	docker compose down

restart: ## Restart container
	docker compose restart

logs: ## View container logs (follow)
	docker compose logs -f

shell: ## Open bash shell in container
	docker exec -it tiktok-scraper /bin/bash

data: ## Show data directory contents
	@echo "=== Videos ==="
	@find data/profiles -name "videos_raw_*.json" 2>/dev/null | wc -l
	@echo "=== Time Series ==="
	@find data/profiles -name "time_series_*.json" 2>/dev/null | wc -l
	@echo "=== Profiles ==="
	@ls -1 data/profiles/ 2>/dev/null | wc -l

stats: ## Show container resource usage
	docker stats tiktok-scraper --no-stream

clean: ## Remove stopped containers and unused images
	docker compose down -v
	docker system prune -f

backup: ## Create backup of all data
	@BACKUP_NAME="backup_$$(date +%Y%m%d_%H%M%S).tar.gz"; \
	tar -czf $$BACKUP_NAME data/ logs/ progress_tracking/ crawl_account/ *.json *.txt; \
	echo "Backup created: $$BACKUP_NAME"

test: ## Run test scrape (first 2 accounts)
	docker exec tiktok-scraper python run_crawler.py --test

status: ## Show scraping progress
	docker exec tiktok-scraper python run_crawler.py --status