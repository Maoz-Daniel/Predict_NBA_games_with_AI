import os
import asyncio
from bs4 import BeautifulSoup  # Help to parse HTML
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout  # Scrape dynamic content
from datetime import datetime  # Handle dates
import time
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


DATA_DIR = os.path.join(PROJECT_ROOT, "data")  # Data directory
SCORES_DIR = os.path.join(DATA_DIR, "scores")  # Scores data
TEST_DIR = os.path.join(DATA_DIR, "test") 

def load_and_clean_data(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(file_path, index_col=None)
    df = df.sort_values(by='date').reset_index(drop=True)
    return df


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

def extract_games_from_date(html, target_date):
    soup = BeautifulSoup(html, 'html.parser')
    games = []
    month = target_date.split("-")[1]
    # i want the first three letters of the name of thedate
    month = datetime.strptime(month, "%m").strftime("%B")[:3].lower()
    year = target_date.split("-")[0]
    day = target_date.split("-")[2]
    find = f"{month} {day}"
    print(find)
    for row in soup.select('tbody tr'):        
        if find in row.text.lower():  # Check if formatted date is in row text
            row  = str(row).split("<")
            teams = ""
            for i in range(len(row)):
                if "teams" in row[i]:
                    split = row[i].split("/")
                    if(teams == ""):
                        teams = split[2]
                    else:
                        games.append((teams , split[2]))

     
    return games

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
        # i want month name
        month = parsed_date.strftime("%B").lower()
        day = parsed_date.day

      
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
        print(url)
        # Fetch HTML content for the month's page
        html = await get_html(url, "#content")
        #print(html)
       
        if not html:
            print("Failed to fetch HTML for the specified month.")
            return
        games = extract_games_from_date(html, date)
        return games
        

    except Exception as e:
        print(f"Error scraping games for {date}: {e}")

def get_win_loss(df ,date , team):
    wins = 0
    losses = 0
    split = date.split("-")
    season = int(split[0])
    if(int(split[1]) >= 10):
        season += 1
    for i in range(len(df)):
        if(df["season"][i] == season and df["Team"][i] == team):
            if(df["won"][i] == 1):
                wins += 1
            else:
                losses += 1
    return wins, losses
            
    
def get_team_stats(date, games):
    df = load_and_clean_data("games_all_clean.csv")  # Just pass the filename
    stats = []
    for game in games:
        team1 = game[0]
        team2 = game[1]
        wins1, losses1 = get_win_loss(df, date, team1)
        wins2, losses2 = get_win_loss(df, date, team2)
        stats.append((team1, wins1, losses1, team2, wins2, losses2))
    return stats



async def get_stats(date):
    games = await scrape_games_for_date(date)
    print("Scraping completed.")
    stats = get_team_stats(date, games)
    return stats



# Run the main function using asyncio
if __name__ == "__main__":
    asyncio.run(get_stats("2025-1-20"))
