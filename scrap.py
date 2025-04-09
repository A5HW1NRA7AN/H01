from search import Search
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd

class Scraper():
    def __init__(self, url):
        self.url = url

    def scrape(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title
            title = soup.title.string if soup.title else 'No Title'

            # Extract all links
            links = [{'href': a['href'], 'text': a.get_text(strip=True)} for a in soup.find_all('a', href=True)]

            # Extract body text
            body = soup.body.get_text(separator=' ', strip=True) if soup.body else 'No Body Content'

            return {
                'title': title,
                'body': body
            }
        except Exception as e:
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    url = "https://www.python.org/"
    scraper = Scraper(url)
    result = scraper.scrape()
    print(result)