import os
import asyncio
from bs4 import BeautifulSoup  # Help to parse HTML
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout  # Scrape dynamic content
import time

# Constants
SEASONS = list(range(2018, 2025))  # 2018-2024 seasons
DATA_DIR = "data"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")  # Standings data
SCORES_DIR = os.path.join(DATA_DIR, "scores")  # Scores data

# Asynchronous function to get HTML content of a page
async def get_html(url, selector, sleep=5, retries=3):
    html = None
    for i in range(1, retries + 1):
        time.sleep(sleep * i)  # Wait for a while before retrying to avoid server blocks

        try:
            async with async_playwright() as p:
                print("Launching browser...")
                browser = await p.chromium.launch()  # Launch a browser
                page = await browser.new_page()  # Open a new page
                await page.goto(url)  # Navigate to the URL
                print(f"Page title: {await page.title()}")  # Print the title of the page
                html = await page.inner_html(selector)  # Get the HTML content of the page

        except PlaywrightTimeout:
            print(f"Timeout error: {url}. Retrying attempt {i}...")
            continue  # Retry again

        else:
            break
    return html

# Main function
async def main():
    season = 2024
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    print("Starting HTML fetch...")
    html = await get_html(url, "#content .filter")  # Get the HTML content of the page
    print("HTML content fetched successfully:")
    print(html)  # Print the HTML content of the page

# Run the main function using asyncio
if __name__ == "__main__":
    asyncio.run(main())
