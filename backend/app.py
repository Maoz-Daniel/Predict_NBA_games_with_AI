import random
from flask import Flask, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import asyncio
from pathlib import Path
import sys
from pathlib import Path

nba_teams = {
    "HAWKS": "ATL",
    "CELTICS": "BOS",
    "NETS": "BRK",
    "HORNETS": "CHO",
    "BULLS": "CHI",
    "CAVALIERS": "CLE",
    "MAVERICKS": "DAL",
    "NUGGETS": "DEN",
    "PISTONS": "DET",
    "WARRIORS": "GSW",
    "ROCKETS": "HOU",
    "PACERS": "IND",
    "CLIPPERS": "LAC",
    "LAKERS": "LAL",
    "GRIZZLIES": "MEM",
    "HEAT": "MIA",
    "BUCKS": "MIL",
    "TIMBERWOLVES": "MIN",
    "PELICANS": "NOP",
    "KNICKS": "NYK",
    "THUNDER": "OKC",
    "MAGIC": "ORL",
    "SIXERS": "PHI",
    "SUNS": "PHO",
    "TRAIL_BLAZERS": "POR",
    "KINGS": "SAC",
    "SPURS": "SAS",
    "RAPTORS": "TOR",
    "JAZZ": "UTA",
    "WIZARDS": "WAS"

}

nba_teams_reverse = {v: k for k, v in nba_teams.items()}

# Add the parent directory to sys.path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Remove the direct process_data imports at the top
# Now import what we need after adding to sys.path
from process_data.get_games import get_stats
# Import the specific function from updated_games instead of the whole module
# from process_data.updated_games import main as update_games_main
from prediction.predictions import predict_games


app = Flask(__name__)
CORS(app)

# Global variable to store the message from the midnight task
midnight_message = {
    "message": "",
    "games": [],
    "gameOfDay": None
}


# Example function to be called at midnight
def midnight_task():

    # games1 = [("TOR", "LAL")]#, ("WAS", "MIL"), ("MIA", "NYK"), ("PHI", "OKC"),("MIN", "HOU"), ("DEN", "SAS"), ("UTA", "CLE"), ("GSW", "DAL"),("SAC", "LAC")]
    # date1="2024-04-02"
    # predict_games(games1,date1)



    global midnight_message
    midnight_message["message"] = f"Midnight task executed at {datetime.now()}"

    
    date = datetime.now().strftime("%Y-%m-%d")

    try:
        # Create event loop and run async function
        stats = asyncio.run(get_stats(date))
        games_to_prediction = []
        for game in stats:
            team1, wins1, losses1, team2, wins2, losses2 = game
            games_to_prediction.append((team1, team2))
        games = []
        predictions = predict_games(games_to_prediction, date)
        for game in stats:
            team1, wins1, losses1, team2, wins2, losses2 = game
            confidence = next((round(c * 100, 2) for t1, t2, c in predictions if t1 == team1 and t2 == team2), None)
            if(confidence > 50):
                prediction = team1
            else:
                prediction = team2
                confidence = 100 - confidence
            print("team1", team1)
            print("team1 reverse", nba_teams_reverse.get(team1, team1))
            print("team2", team2)
            print("team2 reverse", nba_teams_reverse.get(team2, team2))
            game_data = {
                "id": random.randint(1000, 9999),
                "homeTeam": nba_teams_reverse.get(team1, team1),
                "homeRecord": f"{wins1}-{losses1}",
                "awayTeam": nba_teams_reverse.get(team2, team2),
                "awayRecord": f"{wins2}-{losses2}",
                "prediction": nba_teams.get(prediction, prediction),
                "confidence": confidence
            }
            games.append(game_data)
            if games:
                game_of_day = random.choice(games)
                favored_team = game_of_day["prediction"]  # 'DAL', 'NYK', etc.
                confidence = game_of_day["confidence"]  # Confidence percentage
                favored_team = nba_teams_reverse.get(favored_team, favored_team)  # 'Mavericks', 'Knicks', etc.

                if game_of_day["homeTeam"] == favored_team:
                    home_win_chance = confidence
                    away_win_chance = 100 - confidence
                else:
                    home_win_chance = 100 - confidence
                    away_win_chance = confidence

                midnight_message["gameOfDay"] = {
                    "homeTeam": {
                        "name": game_of_day["homeTeam"],
                        "record": game_of_day["homeRecord"],
                        "winChance": round(home_win_chance, 2)
                    },
                    "awayTeam": {
                        "name": game_of_day["awayTeam"],
                        "record": game_of_day["awayRecord"],
                        "winChance": round(away_win_chance, 2)
                    },
                    "gameTime": f"Today, {random.randint(7, 10)}:00 PM ET",
                    "stadium": "NBA Arena"
                }

        midnight_message["games"] = games

    except Exception as e:
        midnight_message["message"] = f"Error: {str(e)}"

    return midnight_message
    

    



scheduler = BackgroundScheduler()
scheduler.add_job(midnight_task, 'interval', seconds=60)  # Run every 30 seconds
# scheduler.add_job(midnight_task, 'cron', hour=0, minute=0)  # Run at midnight
scheduler.start()


@app.route('/api/data', methods=['GET'])
def get_data():
    global midnight_message
    return jsonify(midnight_message)

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
