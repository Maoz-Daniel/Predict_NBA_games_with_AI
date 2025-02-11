import os
import asyncio
from bs4 import BeautifulSoup  # Help to parse HTML
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout  # Scrape dynamic content
from datetime import datetime  # Handle dates
import time
import parse_data as parse_data
import data_cleaning as data_cleaning
import add as add
import pandas as pd

# Constants
DATA_DIR = "data"  # Data directory
SCORES_DIR = os.path.join(DATA_DIR, "scores")  # Scores data
TEST_DIR = "data\\test"  # Test data directory

months = {
    "January": (1, 18, 2025),
}
# Asynchronous function to get HTML content of a page
async def get_html(url, selector, sleep=5, retries=3):
    html = None
    for i in range(1, retries + 1):
        time.sleep(sleep * i)
        try:
            async with async_playwright() as p:
                browser = await p.firefox.launch()
                page = await browser.new_page()
                await page.goto(url)
                html = await page.inner_html(selector)
        except PlaywrightTimeout:
            print(f"Timeout error: {url}. Retrying attempt {i}...")
            continue
        else:
            break
    return html


# Function to scrape games for a specific date
async def scrape_games_for_date(date):
    """
    Scrapes games for a given date.
    :param date: A string in the format 'YYYY-MM-DD'.
    """
    try:
        # Validate and parse the date
        parsed_date = datetime.strptime(date, '%Y-%m-%d')
        year = parsed_date.year
        month = parsed_date.month
        day = parsed_date.day

        # Construct the URL for the specific date  https://www.basketball-reference.com/boxscores/?month=12&day=6&year=2022
        url = f"https://www.basketball-reference.com/boxscores/?month={month}&day={day}&year={year}"
        print(url)
        # Fetch HTML content for the month's page
        html = await get_html(url, "#content")
        #print(html)
       
        if not html:
            print("Failed to fetch HTML for the specified month.")
            return

        # Parse the HTML and extract games for the specific date
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find_all("a")
        hrefs = [l.get("href") for l in links] # Extract the href attribute of the links
        box_scores = [l for l in hrefs if l and "boxscore" in l and ".html" in l] # Filter links containing "boxscores"
        box_scores = [f"https://www.basketball-reference.com{l}" for l in box_scores]  # Add the base URL to the links
        for url in box_scores:
            save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
            if os.path.exists(save_path):
                
                continue
            html = await get_html(url, "#content")
            if not html:
                continue
            with open(save_path, "w+", encoding="utf-8") as f:
                f.write(html)

    except Exception as e:
        print(f"Error scraping games for {date}: {e}")

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, index_col=None)
    df = df.sort_values(by='date').reset_index(drop=True)

    return df

def fix_target_column(date):
    df = load_and_clean_data("data/games_all_clean.csv")
    #i have a row target where its 1 if the team one the next game 0 if they lost and 2 if there is no next game when i add a new game i need to find the previous game and change it from two to wether the team one or not
     # Ensure the date column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    date = pd.to_datetime(date)

    # Filter the DataFrame for the new game(s) on the given date
    new_games = df[df['date'] == date]

    for _, new_game in new_games.iterrows():
        team = new_game['Team']
        outcome = new_game['won']  # Outcome of the new game (1 = win, 0 = loss)
        # Find the team's previous game
        previous_games = df[(df['Team'] == team) & (df['date'] < date)]
        if not previous_games.empty:
            last_game_idx = previous_games['date'].idxmax()  # Get the index of the last game
            # Update the target of the previous game based on the new game's outcome
            df.loc[last_game_idx, 'target'] = 1 if outcome == 1 else 0
    df.to_csv("data/games_all_clean.csv", index=False)
    return 
# Main function
async def main():
    date = "2025-01-24"
    await scrape_games_for_date(date)
    print("Scraping completed.")
    await parse_data.main()
    print("Parsing completed.")
    data_cleaning.main()
    print("Data cleaning completed.")
    add.combine_csv_files()
    print("Data addition completed.")
    fix_target_column(date)
    print("Target column fixed.")


# Run the main function using asyncio
if __name__ == "__main__":
    asyncio.run(main())
