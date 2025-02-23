import aiohttp
import asyncio
import json
import logging
from bs4 import BeautifulSoup
from aiohttp.client_exceptions import ClientError

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

    async def _make_request(self, session, url, retries=0):
        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 429:  # Too Many Requests
                    retry_after = int(response.headers.get("Retry-After", self.retry_delay))
                    self.logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                    if retries < self.max_retries:
                        return await self._make_request(session, url, retries + 1)
        except (ClientError, TimeoutError, asyncio.exceptions.TimeoutError) as e:
            if retries < self.max_retries:
                wait_time = self.retry_delay * (retries + 1)
                self.logger.warning(f"Request failed: {str(e)}. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                return await self._make_request(session, url, retries + 1)
            else:
                self.logger.error(f"Max retries reached for URL: {url}")
        return None

    async def get_total_pages(self, session):
        html = await self._make_request(session, self.base_url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            pagination = soup.select_one(self.selectors["pages"])
            if pagination:
                pages = pagination.text.split()[-1]
                if pages.isdigit():
                    return int(pages)
        return 1

    async def gather_job_urls(self, session, page):
        url = f"{self.base_url}page/{page}/"
        html = await self._make_request(session, url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            listings = soup.select(self.selectors["urls"])
            urls = [listing.get("href") for listing in listings if listing.get("href")]
            self.job_urls.update(urls)

    async def scrape_job_details(self, session, url):
        html = await self._make_request(session, url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            details = soup.select_one(self.selectors["details"])
            description = soup.select_one(self.selectors["description"])

            job_data = clean_job_data(details, description, url)
            if job_data:
                return job_data
        return None

    async def scrape_jobs(self):
        existing_jobs = []
        if self.data_file.exists():
            with open(self.data_file, "r") as f:
                existing_jobs = json.load(f)
        existing_urls = {job["url"] for job in existing_jobs}

        connector = aiohttp.TCPConnector(limit=10, force_close=True)
        timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_read=60)

        async with aiohttp.ClientSession(
            headers=self.headers, cookies=self.cookies, connector=connector, timeout=timeout
        ) as session:
            try:
                total_pages = await self.get_total_pages(session)
                self.logger.info(f"Found {total_pages} pages to scrape")

                for page in range(1, total_pages + 1):
                    self.logger.info(f"Scanning page {page}/{total_pages}")
                    await self.gather_job_urls(session, page)

                    current_urls = self.job_urls
                    if any(url in existing_urls for url in current_urls):
                        self.logger.info("Found existing URL, stopping further scraping")
                        break

                new_urls = [url for url in self.job_urls if url not in existing_urls]
                total_new_urls = len(new_urls)

                if new_urls:
                    self.logger.info(f"Found {total_new_urls} new jobs to scrape")
                    tasks = [self.scrape_job_details(session, url) for url in new_urls]

                    completed = 0
                    job_data = []
                    for task in asyncio.as_completed(tasks):
                        job = await task
                        completed += 1
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
