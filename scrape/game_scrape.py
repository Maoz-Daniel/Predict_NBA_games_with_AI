import os
import asyncio
from bs4 import BeautifulSoup  # Help to parse HTML
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout  # Scrape dynamic content
import time
import web_scrape

# Constants
SEASONS = list(range(2021, 2025))  # 2018-2024 seasons
DATA_DIR = "data" # Data directory
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")  # Standings data
SCORES_DIR = os.path.join(DATA_DIR, "scores")  # Scores data




# Main function
async def scrape_game(standings_file):
    print(f"Reading file: {standings_file}")
    with open(standings_file, 'r') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a")
    hrefs = [l.get("href") for l in links] # Extract the href attribute of the links
    box_scores = [l for l in hrefs if l and "boxscore" in l and ".html" in l] # Filter links containing "boxscores"
    box_scores = [f"https://www.basketball-reference.com{l}" for l in box_scores]  # Add the base URL to the links
    for url in box_scores:
        save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            print(f"File already exists: {save_path}")
            continue
        html = await web_scrape.get_html(url, "#content")
        if not html:
            continue
        with open(save_path, "w+", encoding="utf-8") as f:
            f.write(html)




async def main():
    standings_files = os.listdir(STANDINGS_DIR)
    standings_files = [s for s in standings_files if ".html" in s]
    for f in standings_files:
        filepath = os.path.join(STANDINGS_DIR, f)
        await scrape_game(filepath)


        

# Run the main function using asyncio
if __name__ == "__main__":
    asyncio.run(main())
