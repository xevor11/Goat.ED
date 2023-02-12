import sys
import requests
from bs4 import BeautifulSoup

def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    return soup

def display_progress(current: int, total: int, prefix: str = "", suffix: str = "", decimal_places: int = 1, bar_length: int = 50):
    percentage = (current / total) * 100
    filled_length = int(round(bar_length * current / total))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    sys.stdout.write(f"\r{prefix} |{bar}| {percentage:.{decimal_places}f}% {suffix}")
    if current == total:
        sys.stdout.write("\n")
    sys.stdout.flush()
