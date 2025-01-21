import os
import asyncio
import pandas as pd
from bs4 import BeautifulSoup  # Help to parse HTML
from io import StringIO

SCORE_DIR = "data/scores"  # Scores data
NOT_LINE_SCORE = []



def parse_html(box_score):
    with open(box_score, encoding="utf-8") as f: # Open the file maybe add encoding="utf-8"
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser') #, 'html.parser'
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr.thead")]
    return soup 

def read_line_score(soup):


    try:
        table = soup.find("table", {"id": "line_score"})
        if table is None:
            raise ValueError("Table with id 'line_score' not found in the HTML")


        line_score = pd.read_html(StringIO(str(soup)), attrs={"id": "line_score"}, flavor="lxml")[0]
        cols = list(line_score.columns)
        cols[0] = "Team"
        cols[-1] = "Total"
        line_score.columns = cols
        line_score = line_score[["Team" , "Total"]]
        return line_score
    except Exception as e:
        print(f"Error in read_line_score: {e}")
        return None  # Return None if the table is invalid or missing
    

def read_stats(soup,team, stat):
    df = pd.read_html(StringIO(str(soup)), attrs={"id": f"box-{team}-game-{stat}"}, index_col=0, flavor="lxml")[0]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def read_season_info(soup):
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all("a")]
    season = os.path.basename(hrefs[1]).split("_")[0]
    return season

async def main():
    base_cols = None
    games = []
    box_scores = os.listdir(SCORE_DIR)
    box_scores = [os.path.join(SCORE_DIR, f) for f in box_scores if f.endswith(".html")]
    for box_score in box_scores:
        try:
            soup = parse_html(box_score)
            line_score = read_line_score(soup)
            teams = list(line_score["Team"])
            
            summeries = []
            for team in teams:
                basic = read_stats(soup, team, "basic")
                advanced = read_stats(soup, team, "advanced")
                totals = pd.concat([basic.iloc[-1,:], advanced.iloc[-1,:]])  
                totals.index = totals.index.str.lower()
                maxes = pd.concat([basic.iloc[:-1,:].max(), advanced.iloc[:-1,:].max()])
                maxes.index = maxes.index.str.lower() + "_max"
                summery = pd.concat([totals, maxes])
                if base_cols is None:
                    base_cols = list(summery.index.drop_duplicates(keep="first"))
                    base_cols = [b for b in base_cols if "bpm" not in b]
                summery = summery[base_cols]
                summeries.append(summery)
            summery = pd.concat(summeries, axis=1).T
            game = pd.concat([summery, line_score], axis=1)
            
            game["home"] = [0,1]
            game_opp = game.iloc[::-1].reset_index()
            game_opp.columns += "_opp"
            full_game = pd.concat([game, game_opp], axis=1)
            
            full_game["season"] = read_season_info(soup)
            full_game["date"] = os.path.basename(box_score)[:8]
            full_game["date"] = pd.to_datetime(full_game["date"], format="%Y%m%d")
            full_game["won"] = full_game["Total"] > full_game["Total_opp"]
            games.append(full_game)
            if len(games) % 50 == 0:
                print(f"{(len(games) / len(box_scores)) * 100:.2f}% done")
        except Exception as e:
            # Print the date of the game that caused the error
            print(f"Cnnot parse {box_score} due to error: {e} with date {os.path.basename(box_score)[:8]}")
    if(games):
        games_df = pd.concat(games, ignore_index=True)
        games_df.to_csv("data/newGames.csv", index=False)
        #erase from data/scores
        for file in box_scores:
            os.remove(file)
    


        

# Run the main function using asyncio
if __name__ == "__main__":
    asyncio.run(main())
