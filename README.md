# job-scraper/job-scraper/README.md

# Job Scraper

This project is a web scraping application that uses Playwright to scrape job data from [Khaleej Times Jobs](https://buzzon.khaleejtimes.com/ad-category/jobs-vacancies/). The scraper is designed to run daily and store the scraped job data in a JSON file.

## Project Structure

```
job-scraper
├── src
│   ├── main.py                # Entry point of the application
│   ├── scraper
│   │   ├── __init__.py        # Marks the scraper directory as a package
│   │   └── scraper.py         # Contains the JobScraper class
│   ├── config
│   │   ├── __init__.py        # Marks the config directory as a package
│   │   └── config.py          # Configuration settings for the scraper
│   └── utils
│       ├── __init__.py        # Marks the utils directory as a package
│       └── helpers.py         # Utility functions for data processing
├── tests
│   ├── __init__.py            # Marks the tests directory as a package
│   └── test_scraper.py        # Unit tests for the JobScraper class
├── data
│   └── jobs.json              # JSON file to store scraped job data
├── requirements.txt            # Lists project dependencies
├── .gitignore                  # Specifies files to ignore by Git
└── README.md                   # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd job-scraper
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the scraper:
   ```
   python src/main.py
   ```

## Usage

The scraper will automatically fetch job listings from the specified URL and save the data to `data/jobs.json`. You can schedule this script to run daily using a task scheduler like cron (Linux) or Task Scheduler (Windows).

## License

This project is licensed under the MIT License.