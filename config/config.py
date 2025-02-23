from pathlib import Path


class Config:
    BASE_URL = "https://buzzon.khaleejtimes.com/ad-category/jobs-vacancies/"

    DATA_FILE = Path("data/jobs.json")

    SELECTORS = {
        "pages": "div.paging > div.pages > span.total",
        "urls": "div.post-left > a",
        "description": "div.single-main",
        "details": "div.bigright ul",
    }

    SCHEDULER_INTERVAL = 24

    HEADERS = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "max-age=0",
        "sec-ch-ua": '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "cross-site",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    }

    COOKIES = {
        "ct_sfw_pass_key": "4a1b84682f291a73f6cf5f3c3f2590810",
        "wordpress_test_cookie": "WP Cookie check",
        "ct_timezone": "5",
        "apbct_headless": "false",
        "ct_checked_emails": "0",
        "ct_checkjs": "2d37305e2dd89d174ca6cd5f7c0cfcab59ba890e5040b9625d8a5cd5e09ad724",
        "device_uuid": "91df5b4b-4985-459c-9b46-a0f83c124a44",
        "ct_mouse_moved": "true",
        "ct_has_scrolled": "true",
        "PHPSESSID": "tej3q9gt9e26bt8nqpu1s49ubv",
    }
