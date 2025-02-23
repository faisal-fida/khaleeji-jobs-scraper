import schedule
import time
from datetime import datetime
import logging
from scraper.scraper import JobScraper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_scraper():
    logger.info("Starting scheduled job scraping...")
    try:
        scraper = JobScraper()
        scraper.scrape_jobs()
        logger.info(f"Scheduled scraping completed at {datetime.now()}")
    except Exception as e:
        logger.error(f"Error in scheduled scraping: {str(e)}")


def main():
    logger.info("Job scraper scheduler starting...")

    # Schedule the scraper to run every 24 hours at 00:00
    schedule.every().day.at("00:00").do(run_scraper)

    run_scraper()

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check schedule every minute


if __name__ == "__main__":
    main()
