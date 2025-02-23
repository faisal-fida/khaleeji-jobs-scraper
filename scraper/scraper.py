import requests
import json
import logging
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException
from time import sleep

from utils.helpers import clean_job_data
from config.config import Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class JobScraper:
    def __init__(self):
        self.base_url = Config.BASE_URL
        self.data_file = Config.DATA_FILE
        self.headers = Config.HEADERS
        self.cookies = Config.COOKIES
        self.selectors = Config.SELECTORS
        self.job_urls = set()
        self.max_retries = 5
        self.retry_delay = 8
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.cookies.update(self.cookies)

    def _make_request(self, url, retries=0):
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 429:  # Too Many Requests
                retry_after = int(response.headers.get("Retry-After", self.retry_delay))
                self.logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                sleep(retry_after)
                if retries < self.max_retries:
                    return self._make_request(url, retries + 1)
        except RequestException as e:
            if retries < self.max_retries:
                wait_time = self.retry_delay * (retries + 1)
                self.logger.warning(f"Request failed: {str(e)}. Retrying in {wait_time} seconds...")
                sleep(wait_time)
                return self._make_request(url, retries + 1)
            else:
                self.logger.error(f"Max retries reached for URL: {url}")
        return None

    def get_total_pages(self):
        html = self._make_request(self.base_url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            pagination = soup.select_one(self.selectors["pages"])
            if pagination:
                pages = pagination.text.split()[-1]
                if pages.isdigit():
                    return int(pages)
        return 1

    def gather_job_urls(self, page):
        url = f"{self.base_url}page/{page}/"
        html = self._make_request(url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            listings = soup.select(self.selectors["urls"])
            urls = [listing.get("href") for listing in listings if listing.get("href")]
            self.job_urls.update(urls)

    def scrape_job_details(self, url):
        html = self._make_request(url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            details = soup.select_one(self.selectors["details"])
            description = soup.select_one(self.selectors["description"])

            job_data = clean_job_data(details, description, url)
            if job_data:
                return job_data
        return None

    def scrape_jobs(self):
        existing_jobs = []
        if self.data_file.exists():
            with open(self.data_file, "r") as f:
                existing_jobs = json.load(f)
        existing_urls = {job["url"] for job in existing_jobs}

        try:
            total_pages = self.get_total_pages()
            self.logger.info(f"Found {total_pages} pages to scrape")

            # Gather job URLs using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=10) as executor:
                for page in range(1, total_pages + 1):
                    self.logger.info(f"Scanning page {page}/{total_pages}")
                    executor.submit(self.gather_job_urls, page)

                    current_urls = self.job_urls
                    if any(url in existing_urls for url in current_urls):
                        self.logger.info("Found existing URL, stopping further scraping")
                        break

            new_urls = [url for url in self.job_urls if url not in existing_urls]
            total_new_urls = len(new_urls)

            if new_urls:
                self.logger.info(f"Found {total_new_urls} new jobs to scrape")
                job_data = []
                completed = 0

                # Scrape job details using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_url = {
                        executor.submit(self.scrape_job_details, url): url for url in new_urls
                    }

                    for future in as_completed(future_to_url):
                        completed += 1
                        job = future.result()
                        if job:
                            job_data.append(job)
                        self.logger.info(f"Progress: {completed}/{total_new_urls} jobs scraped")

                job_data = [job for job in job_data if job]  # Remove None values
                self.logger.info(f"Successfully scraped {len(job_data)} jobs")

                all_jobs = existing_jobs + job_data
                with open(self.data_file, "w") as f:
                    json.dump(all_jobs, f, indent=2)
                self.logger.info(f"Total jobs in database: {len(all_jobs)}")
            else:
                self.logger.info("No new jobs to scrape")

        except Exception as e:
            self.logger.error(f"Error occurred: {str(e)}")
