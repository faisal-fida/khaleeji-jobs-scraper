from scraper.scraper import JobScraper


def main():
    scraper = JobScraper()
    scraper.scrape_jobs()


if __name__ == "__main__":
    main()
