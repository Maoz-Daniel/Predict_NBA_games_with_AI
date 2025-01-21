from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from time import sleep
import os
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")  # Data directory



def calculate_rolling_means(df, columns, window=10):
    """
    Calculate rolling means for specified columns over a given window.
    """
    for col in columns:
        df[f"{col}rolling{window}"] = df.groupby("Team")[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
    print("The new columns are:", df.columns)
    return df

def calculate_win_percentage(df, window=10):
    """
    Calculate rolling win percentage over a given window.
    """
    df['win_percentage_rolling'] = df.groupby("Team")['won'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    return df

def evaluate_model(data, model, predictors):
    """
    Evaluate the model on a holdout test set.
    """
    train = data[data['season'] < data['season'].max()]
    test = data[data['season'] == data['season'].max()]

    model.fit(train[predictors], train['won'])
    preds = model.predict(test[predictors])
    probas = model.predict_proba(test[predictors])[:, 1]

    accuracy = metrics.accuracy_score(test['won'], preds)
    f1 = metrics.f1_score(test['won'], preds, average="weighted")
    roc_auc = metrics.roc_auc_score(test['won'], probas)
    return accuracy, f1, roc_auc

def calculate_days_rest(df):
    """
    Calculate days of rest for each team between games.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['days_rest'] = df.groupby('Team')['date'].diff().dt.days.fillna(0).astype(int)
    return df
    
def calculate_rolling_avg(df, columns, date, team, window=10):
    """
    Calculate rolling averages for specified columns for a given team up to a certain date.

    Args:
        df (pd.DataFrame): The dataset containing game data.
        columns (list): List of column names to compute rolling averages for.
        date (str): The game date in 'YYYY-MM-DD' format.
        team (str): The team name.
        window (int): The number of previous games to consider for the rolling average.

    Returns:
        dict: A dictionary containing rolling averages for the specified columns.
    """
    # Convert date to datetime
    date = pd.to_datetime(date)

    # Filter the dataset for the specified team and past games before the given date
    past_games = df[(df['Team'] == team) & (df['date'] < date)].tail(window)

    if past_games.empty:
        raise ValueError(f"Not enough past games found for team {team} before {date}")

    # Compute rolling averages for specified columns
    rolling_avg = past_games[columns].mean().to_dict()
 
    return rolling_avg



def predict(df, model, date, games, window=10, pca=None, scaler=None):
    predictions = []
    date = pd.to_datetime(date)

    # First, calculate all the rolling means for the entire dataset
    rolling_columns = [
        "fg%", "3p%", "ft%", "drb", "ast", "stl", "blk", "tov", "pts", "orb",
        "ts%", "usg%", 'ft%_opp', 'ast_opp', 'drb_opp', '3p%_opp', 'fg%_opp', "orb_opp"
    ]
    
    # Define which stats are positive/negative
    positive_stats = {"fg%", "3p%", "ft%", "drb", "ast", "stl", "blk", "pts", "orb", "ts%", "usg%"}
    negative_stats = {"tov", "ft%_opp", "ast_opp", "drb_opp", "3p%_opp", "fg%_opp", "orb_opp"}
    
    df = calculate_rolling_means(df, rolling_columns, window=10)
    df = calculate_win_percentage(df, window=10)
    df = calculate_days_rest(df)
    
    modeling_columns = [f"{col}rolling10" for col in rolling_columns] + ["win_percentage_rolling", "days_rest"]

    for home_team, away_team in games:
        try:
            for i in range(2):
                home_team, away_team = away_team, home_team  # Swap home and away teams
                # Get the most recent data for each team
                home_data = df[df['Team'] == home_team].sort_values('date').tail(1)
                away_data = df[df['Team'] == away_team].sort_values('date').tail(1)
                
                if home_data.empty or away_data.empty:
                    print(f"No data found for {home_team} or {away_team}")
                    continue
                    
                # Create feature vector with proper stat interpretation
                feature_vector = {}
                for col in modeling_columns:
                    base_col = col.replace('rolling10', '')
                    home_val = home_data[col].iloc[0]
                    away_val = away_data[col].iloc[0]
                    
                    # For positive stats, higher is better
                    if base_col in positive_stats:
                        feature_vector[col] = float(home_val) - float(away_val)
                    # For negative stats, lower is better
                    elif base_col in negative_stats:
                        feature_vector[col] = float(away_val) - float(home_val)
                    # For other stats like win_percentage and days_rest
                    else:
                        feature_vector[col] = float(home_val) - float(away_val)
                
                feature_vector_df = pd.DataFrame([feature_vector])
                
                if scaler:
                    feature_vector_scaled = scaler.transform(feature_vector_df)
                    feature_vector_scaled_df = pd.DataFrame(feature_vector_scaled, columns=feature_vector_df.columns)
                else:
                    feature_vector_scaled_df = feature_vector_df

                prediction_df = feature_vector_scaled_df

                # Make prediction
                pred = model.predict(prediction_df)[0]
                prob = model.predict_proba(prediction_df)[0]
                
                result = "Win" if pred == 1 else "Loss"
                confidence = max(prob)
                win_prob = prob[1]  # Probability of home team winning
                
                if i == 0:
                    win_prob_opp = win_prob
                    confidence_opp = confidence
                else:
                    print(f"{home_team} vs {away_team}: {win_prob} ,{win_prob_opp}")
                    if win_prob_opp > win_prob:
                        predictions.append({
                            "home_team": home_team,
                            "away_team": away_team,
                            "date": date.date(),
                            "prediction": "Loss",
                            "confidence": confidence_opp,
                        })
                    else:
                        predictions.append({
                            "home_team": home_team,
                            "away_team": away_team,
                            "date": date.date(),
                            "prediction": "Win",
                            "confidence": confidence,
                        })

        except Exception as e:
            print(f"Error processing {home_team} vs {away_team}: {e}")
            continue

    return predictions

    
    

def predict_games(games, date):
    upd
    file_path = os.path.join(DATA_DIR, "games_all_clean.csv")
    df = pd.read_csv(file_path, index_col=None)

    # Define important stats with weights
    rolling_columns = [
        "fg%", "3p%", "ft%", "drb", "ast", "stl", "blk", "tov", "pts", "orb",
        "ts%", "usg%", 'ft%_opp', 'ast_opp', 'drb_opp', '3p%_opp', 'fg%_opp', "orb_opp"
    ]
    
    # Calculate features
    df = calculate_rolling_means(df, rolling_columns, window=10)
    df = calculate_win_percentage(df, window=10)
    df = calculate_days_rest(df)
    
    modeling_columns = [f"{col}rolling10" for col in rolling_columns] + ["win_percentage_rolling", "days_rest"]
    
    # Use StandardScaler instead of MinMaxScaler
    scaler = StandardScaler()
    df[modeling_columns] = scaler.fit_transform(df[modeling_columns])

    # Train model on the scaled features directly
    train = df[df['season'] < df['season'].max()]
    test = df[df['season'] == df['season'].max()]
    
    # Modified RandomForest parameters for more sensitivity
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        class_weight={0: 1, 1: 1.2}  # Slight boost to home wins
    )
    
    # Train the model
    model.fit(train[modeling_columns], train['won'])
    
    # Evaluate
    preds = model.predict(test[modeling_columns])
    accuracy = metrics.accuracy_score(test['won'], preds)
    f1 = metrics.f1_score(test['won'], preds)
    roc_auc = metrics.roc_auc_score(test['won'], model.predict_proba(test[modeling_columns])[:, 1])
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': modeling_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 5 most important features:")
    print(importances.head())
    
    # Make predictions
    file_path = os.path.join(DATA_DIR, "games_all_clean.csv")
    redo_df = pd.read_csv(file_path, index_col=None)
    predictions = predict(redo_df, model, date, games, scaler=scaler)
    prediction = []
    for pred in predictions:
        prediction.append((pred["home_team"], pred["away_team"], float(pred["confidence"]) if pred["prediction"] == "Win" else float(1 - pred["confidence"])))
    print(prediction)
    return prediction


if __name__ == "__main__":
    games = [("LAL", "TOR"), ("WAS", "MIL"), ("MIA", "NYK"), ("PHI", "OKC"),("MIN", "HOU"), ("SAS", "DEN"), ("CLE", "UTA"), ("GSW", "DAL"),("LAC", "SAC")]
    date="2024-04-02"
    predict_games(games,date)