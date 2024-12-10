import os
import asyncio
from bs4 import BeautifulSoup  # Help to parse HTML
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout  # Scrape dynamic content
import time

# Constants
SEASONS = list(range(2021, 2025))  # 2018-2024 seasons
DATA_DIR = "data" # Data directory
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")  # Standings data
SCORES_DIR = os.path.join(DATA_DIR, "scores")  # Scores data


# Asynchronous function to get HTML content of a page
async def get_html(url, selector, sleep=5, retries=3):
    html = None
    for i in range(1, retries + 1): # Retry up to 3 times
        time.sleep(sleep * i)  # Wait for a while before retrying to avoid server blocks
        try:
            async with async_playwright() as p: # Create a Playwright instance
                print("Launching browser...")
                browser = await p.firefox.launch()  # Launch a browser
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

async def scrape_season(season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html" # URL of the page to scrape
    html = await get_html(url, "#content .filter")  # Get the HTML content of the page
    print("Starting HTML fetch...")
    print("HTML content fetched successfully:")
    soup = BeautifulSoup(html)  # Parse the HTML content using BeautifulSoup
    links = soup.find_all("a")  # Find all links in the HTML content
    href = [l["href"] for l in links]  # Extract the href attribute of the links
    standing_pages = [f"https://basketball-reference.com{l}" for l in href]  # Filter links containing "standings"
    print(f"Found {len(standing_pages)} standing pages for season {season}")
    for url in standing_pages:
        print("Trying to split URL")
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])  # Save path for the standings data
        print(f"Saving file to: {save_path}")
        if os.path.exists(save_path):
            print(f"File already exists: {save_path}")
            continue

        html = await get_html(url, "#all_schedule")
        with open(save_path, "w+") as f:
            f.write(html)

# Main function
async def main():
    for season in SEASONS:
        await scrape_season(season)


# Run the main function using asyncio
if __name__ == "__main__":
    asyncio.run(main())
