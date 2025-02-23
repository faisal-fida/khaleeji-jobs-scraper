from datetime import datetime


def clean_job_data(details, description, url):
    description = description.get_text(strip=True) if description else None
    details = details if details else None

    job_data = {"url": url, "scraped_at": datetime.now().isoformat(), "description": description}

    for item in details.find_all("li"):
        label_elem = item.find("span")
        if not label_elem:
            continue

        label = label_elem.text.strip().rstrip(":").lower()

        value = item.get_text(strip=True).replace(label_elem.get_text(strip=True), "").strip()

        field_name = label.replace(" ", "_")

        if field_name == "email":
            email_link = item.find("a")
            if email_link:
                value = email_link["href"].replace("mailto:", "")

        job_data[field_name] = value

    return job_data
