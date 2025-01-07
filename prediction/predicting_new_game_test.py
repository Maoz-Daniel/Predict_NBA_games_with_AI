import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from datetime import timedelta

teams_to_num = {
  'GSW': 0,
  'HOU': 1,
  'BOS': 2,
  'CLE': 3,
  'MIL': 4,
  'SAS': 5,
  'BRK': 6,
  'PHO': 7,
  'NOP': 8,
  'ATL': 9,
  'ORL': 10,
  'UTA': 11,
  'MIA': 12,
  'DAL': 13,
  'WAS': 14,
  'SAC': 15,
  'MEM': 16,
  'POR': 17,
  'PHI': 18,
  'DET': 19,
  'DEN': 20,
  'CHO': 21,
  'IND': 22,
  'MIN': 23,
  'LAC': 24,
  'TOR': 25,
  'OKC': 26,
  'CHI': 27,
  'NYK': 28,
  'LAL': 29
}

def save_data(df, output_path):
    """
    Save the cleaned dataset to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

def predict_upcoming_game(
    model, 
    scaler, 
    df_preprocessed,
    date,
    home,
    team,
    team_opp,
    season,
    # The stats that you'd typically know ahead of time go here:
    # For demonstration, I'm including them, but you might need to decide
    # how to handle unknown or partially known stats:
    fg_max=0, fga_max=0, 
    three_p_max=0, three_pa_max=0,
    ft_max=0, fta_max=0,
    fg_max_opp=0, fga_max_opp=0,
    three_p_max_opp=0, three_pa_max_opp=0,
    ft_max_opp=0, fta_max_opp=0
):
    """
    Predict the outcome of an upcoming game using the SAME model and SAME pipeline.
    Assumes you have a 'df_preprocessed' that's the same format returned by 'preprocess_data'.
    
    Parameters:
    -----------
    - model: The trained XGBClassifier
    - scaler: The fitted StandardScaler
    - df_preprocessed: The full preprocessed DataFrame (with historical data)
    - date: The date (as a string or datetime) of the upcoming match
    - home: 1 if the "team" is home, else 0
    - team: string name of the team
    - team_opp: string name of the opponent
    - season: The season integer (e.g. 2024)
    - all the other features that your model expects for the new game
    """
    
    # 1) Convert date to datetime if needed
    date = pd.to_datetime(date)
    
    # 2) Build a synthetic row of data that matches your 'df' columns before final feature selection
    #    We'll store 'won' as None or -1 (some placeholder), because obviously we don't know it.
    synthetic_row = {
        'date': date,
        'home': home,
        'Team': team,  # Careful: your 'df' expects final cat.codes, but let's put the raw strings for now
        'Team_opp': team_opp,
        'season': season,
        'fg_max': fg_max,
        'fga_max': fga_max,
        '3p_max': three_p_max,
        '3pa_max': three_pa_max,
        'ft_max': ft_max,
        'fta_max': fta_max,
        'fg_max_opp': fg_max_opp,
        'fga_max_opp': fga_max_opp,
        '3p_max_opp': three_p_max_opp,
        '3pa_max_opp': three_pa_max_opp,
        'ft_max_opp': ft_max_opp,
        'fta_max_opp': fta_max_opp,
        'won': 0  # placeholder, won't matter for prediction
    }
    
    # 3) Append to the original preprocessed df (BEFORE feature selection) 
    #    so we can re-run or re-check the group-based calculations.
    #    But if your 'df_preprocessed' is already stripped to final columns, 
    #    you might need your "raw" df from before dropping columns.
    temp_df = df_preprocessed.copy()
    new_row_df = pd.DataFrame([synthetic_row])
    temp_df = pd.concat([temp_df, new_row_df], ignore_index=True)
    
    # 4) Re-run whatever logic is needed to update 'days_rest', 'win_rate_5', etc.
    #    This depends on your pipeline. If you rely on 'preprocess_data' or 'add_team_stats',
    #    you might need to call them again. 
    #    However, 'preprocess_data' expects a CSV path; 
    #    you might factor out the code to do "just the transformations".
    
    #    Example approach:
    #    (a) re-sort and add team stats again. 
    #    (b) or, if you keep the final columns only in df_preprocessed, you’ll have to 
    #        do partial calculations yourself.
    
    # For demonstration, let's say we do something simpler: 
    # we'll just fill in some default or last-known values for the appended row 
    # (You might prefer a more sophisticated approach.)
    
    # Sort by date (our new row is presumably the last date).
    temp_df = temp_df.sort_values('date').reset_index(drop=True)

    
    # 5) Now, re-convert the Teams to cat.codes so the new row has the correct numeric codes
    #    The best way is to re-use the SAME category mapping that was used originally.
    #    If your code used: df['Team'].astype('category').cat.codes,
    #    We can replicate that on 'temp_df'.
    
    # Guarantee consistent categories for 'Team' and 'Team_opp'
    # For that, we might store the original categories from df_preprocessed.
    # Example:
    team_cats = df_preprocessed['Team'].astype('category').cat.categories
    team_opp_cats = df_preprocessed['Team_opp'].astype('category').cat.categories
    
    temp_df['Team'] = pd.Categorical(temp_df['Team'], categories=team_cats).codes
    temp_df['Team_opp'] = pd.Categorical(temp_df['Team_opp'], categories=team_opp_cats).codes
    
    # 6) Fill any newly added columns or calculations that your final model expects.
    #    If your final feature set is:
    final_features = [
        'home', 'Team', 'Team_opp',
        'fg_max', 'fga_max', '3p_max', '3pa_max', 'ft_max', 'fta_max',
        'fg_max_opp', 'fga_max_opp', '3p_max_opp', '3pa_max_opp', 'ft_max_opp', 'fta_max_opp',
        'win_rate_5', 'season_win_rate',
        'opp_win_rate_5', 'opp_season_win_rate',
        'days_rest', 'opp_days_rest'
    ]
    
    # If your newly appended row is missing `win_rate_5`, etc., either you recalculate them 
    # or fallback to default (like 0.5 or from the last known). 
    # For demonstration:
    # (In a real scenario, you'd call your add_team_stats function on the entire df again and recast.)
    temp_df['win_rate_5'] = temp_df['win_rate_5'].fillna(0.5)
    temp_df['season_win_rate'] = temp_df['season_win_rate'].fillna(0.5)
    temp_df['opp_win_rate_5'] = temp_df['opp_win_rate_5'].fillna(0.5)
    temp_df['opp_season_win_rate'] = temp_df['opp_season_win_rate'].fillna(0.5)
    temp_df['days_rest'] = temp_df['days_rest'].fillna(3)
    temp_df['opp_days_rest'] = temp_df['opp_days_rest'].fillna(3)
    
    # 7) Isolate the last row (the synthetic upcoming game), 
    #    dropping [won, season, date, etc.] that model doesn't need
    #    or that we won't feed to the model.
    row_to_predict = temp_df.iloc[-1][final_features]
    
    # 8) Scale this single-row input using your already-fit scaler
    row_scaled = scaler.transform([row_to_predict])
    
    # 9) Predict
    prediction = model.predict(row_scaled)[0]
    prediction_proba = model.predict_proba(row_scaled)[0][1]  # Probability of "win" class
    
    # Return both the label (0 or 1) and the probability
    return prediction, prediction_proba



def add_team_stats(df):
    """Add historical statistics for each team."""
    df = df.sort_values('date')  # Sort by date
    
    teams = df['Team'].unique()
    team_stats = {team: {
        'last_5_wins': 0,
        'season_wins': 0,
        'season_games': 0
    } for team in teams}
    
    new_rows = []
    for idx, row in df.iterrows():
        team = row['Team']
        team_opp = row['Team_opp']
        season = row['season']
        
        # Add current stats to the row
        new_row = row.copy()
        new_row['win_rate_5'] = team_stats[team]['last_5_wins'] / 5 if team_stats[team]['season_games'] >= 5 else 0.5
        new_row['season_win_rate'] = (team_stats[team]['season_wins'] / team_stats[team]['season_games'] 
                                    if team_stats[team]['season_games'] > 0 else 0.5)
        new_row['opp_win_rate_5'] = team_stats[team_opp]['last_5_wins'] / 5 if team_stats[team_opp]['season_games'] >= 5 else 0.5
        new_row['opp_season_win_rate'] = (team_stats[team_opp]['season_wins'] / team_stats[team_opp]['season_games']
                                        if team_stats[team_opp]['season_games'] > 0 else 0.5)
        
        new_rows.append(new_row)
        
        # Update statistics after the game
        won = row['won']
        team_stats[team]['season_games'] += 1
        team_stats[team]['season_wins'] += won
        
        if team_stats[team]['season_games'] >= 5:
            team_stats[team]['last_5_wins'] = (team_stats[team]['last_5_wins'] * 4 + won) / 5
        else:
            team_stats[team]['last_5_wins'] = won
            
        # Reset stats for new season
        if idx > 0 and df.iloc[idx-1]['season'] != season:
            team_stats[team] = {
                'last_5_wins': 0,
                'season_wins': 0,
                'season_games': 0
            }
    
    return pd.DataFrame(new_rows)

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Process date
    df['date'] = pd.to_datetime(df['date'])
    
    # Add historical stats
    df = add_team_stats(df)
    
    # Add additional features
    df['days_rest'] = df.groupby('Team')['date'].diff().dt.days.fillna(3)
    df['opp_days_rest'] = df.groupby('Team_opp')['date'].diff().dt.days.fillna(3)
    
    # Final feature selection
    features = [
        'home', 'Team', 'Team_opp',
        'fg_max', 'fga_max', '3p_max', '3pa_max', 'ft_max', 'fta_max',
        'fg_max_opp', 'fga_max_opp', '3p_max_opp', '3pa_max_opp', 'ft_max_opp', 'fta_max_opp',
        'win_rate_5', 'season_win_rate',
        'opp_win_rate_5', 'opp_season_win_rate',
        'days_rest', 'opp_days_rest'
    ]
    unique_teams = df['Team'].unique()
    print(unique_teams)
    # Convert categorical teams to numeric codes
    # Replace the 'Team' column with your global integer codes
    df['Team'] = df['Team'].map(teams_to_num)

    # Replace the 'Team_opp' column with the same dictionary
    df['Team_opp'] = df['Team_opp'].map(teams_to_num)

    
    return df[features + ['won', 'season']]

def train_and_evaluate(file_path):
    df = preprocess_data(file_path)
    
    # Split data by seasons
    train_seasons = df['season'] < df['season'].max()
    X_train = df[train_seasons].drop(['won', 'season'], axis=1)
    y_train = df[train_seasons]['won']
    X_test = df[~train_seasons].drop(['won', 'season'], axis=1)
    y_test = df[~train_seasons]['won']
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the XGBoost model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    print(f"Model Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nPerformance Report:")
    print(classification_report(y_test, predictions))
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Return the important objects for later use
    return model, scaler, df
    df = preprocess_data(file_path)
    
    # Split data by seasons
    train_seasons = df['season'] < df['season'].max()
    X_train = df[train_seasons].drop(['won', 'season'], axis=1)
    y_train = df[train_seasons]['won']
    X_test = df[~train_seasons].drop(['won', 'season'], axis=1)
    y_test = df[~train_seasons]['won']
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the XGBoost model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    print(f"Model Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nPerformance Report:")
    print(classification_report(y_test, predictions))
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)




def get_last_n_games_average(df, team_name, season, feature, n=10):
    team_df = df[(df['Team'] == team_name) & (df['season'] == season)]
    if team_df.empty:  # No rows found at all for this team/season
        print(f"No team found for {team_name} in season {season} and feature {feature}")
        return None
    # Sort descending by date so the most recent games come first
    #team_df = team_df.sort_values(by='date', ascending=False)
    
    # 'head(n)' will give up to n games. If the team has fewer than n games,
    # it will just return however many exist. If that is 0, it's truly empty.
    recent_games = team_df.head(n)
    if recent_games.empty:  # Could happen if the team_df had 0 rows after filtering
        print(f"No games found for {team_name} in season {season} and feature {feature}")
        return None
    
    # Return the average of the specified feature column
    return recent_games[feature].mean()



def safe_val(val, default=0):
    return val if val is not None else default




# Run the enhanced model
file_path = "data\\first_half_2024.csv"
model, scaler, df_preprocessed = train_and_evaluate(file_path)

home_team = "DEN"
opp_team = "PHO"
save_data(df_preprocessed,"data\\temp.csv")
team = teams_to_num[home_team]
avg_fg_max_tor = get_last_n_games_average(df_preprocessed, team, 2024, "fg_max", n=10)
avg_fga_max_tor = get_last_n_games_average(df_preprocessed, team, 2024, "fga_max", n=10)
avg_3p_max_tor = get_last_n_games_average(df_preprocessed, team, 2024, "3p_max", n=10)
avg_3pa_max_tor = get_last_n_games_average(df_preprocessed, team, 2024, "3pa_max", n=10)
avg_ft_max_tor = get_last_n_games_average(df_preprocessed, team, 2024, "ft_max", n=10)
avg_fta_max_tor = get_last_n_games_average(df_preprocessed, team, 2024, "fta_max", n=10)

opp_team =teams_to_num[opp_team]
avg_fg_max_nop = get_last_n_games_average(df_preprocessed, opp_team, 2024, "fg_max", n=10)
avg_fga_max_nop = get_last_n_games_average(df_preprocessed, opp_team, 2024, "fga_max", n=10)
avg_3p_max_nop = get_last_n_games_average(df_preprocessed, opp_team, 2024, "3p_max", n=10)
avg_3pa_max_nop = get_last_n_games_average(df_preprocessed, opp_team, 2024, "3pa_max", n=10)
avg_ft_max_nop = get_last_n_games_average(df_preprocessed, opp_team, 2024, "ft_max", n=10)
avg_fta_max_nop = get_last_n_games_average(df_preprocessed, opp_team, 2024, "fta_max", n=10)

print(f"{home_team} averages:", avg_fg_max_tor, avg_fga_max_tor, avg_3p_max_tor, avg_3pa_max_tor, avg_ft_max_tor, avg_fta_max_tor)
print(f"{opp_team} averages:", avg_fg_max_nop, avg_fga_max_nop, avg_3p_max_nop, avg_3pa_max_nop, avg_ft_max_nop, avg_fta_max_nop)


avg_fg_max_tor = safe_val(avg_fg_max_tor)
avg_fga_max_tor = safe_val(avg_fga_max_tor)
avg_3p_max_tor  = safe_val(avg_3p_max_tor)
avg_3pa_max_tor = safe_val(avg_3pa_max_tor)
avg_ft_max_tor  = safe_val(avg_ft_max_tor)
avg_fta_max_tor = safe_val(avg_fta_max_tor)

avg_fg_max_nop = safe_val(avg_fg_max_nop)
avg_fga_max_nop = safe_val(avg_fga_max_nop)
avg_3p_max_nop  = safe_val(avg_3p_max_nop)
avg_3pa_max_nop = safe_val(avg_3pa_max_nop)
avg_ft_max_nop  = safe_val(avg_ft_max_nop)
avg_fta_max_nop = safe_val(avg_fta_max_nop)

print(f"{home_team} averages:", avg_fg_max_tor, avg_fga_max_tor, avg_3p_max_tor, avg_3pa_max_tor, avg_ft_max_tor, avg_fta_max_tor)
print(f"{opp_team} averages:", avg_fg_max_nop, avg_fga_max_nop, avg_3p_max_nop, avg_3pa_max_nop, avg_ft_max_nop, avg_fta_max_nop)

pred_label, pred_prob = predict_upcoming_game(
    model=model,
    scaler=scaler,
    df_preprocessed=df_preprocessed,
    date="2023-03-05",
    home=1,           # Suppose the "team" is playing at home
    team= home_team,    # Must match the original categories
    team_opp= opp_team,  # Must match the original categories
    season=2023,
    fg_max=23, fga_max=23,   # Because we don't know them yet
    three_p_max=23, three_pa_max=24,
    ft_max=222, fta_max=987,
    fg_max_opp=34, fga_max_opp=43,
    three_p_max_opp=433, three_pa_max_opp=77,
    ft_max_opp=6, fta_max_opp=666
)

print("Predicted label:", pred_label)
print("Probability of winning:", pred_prob)

pred_label, pred_prob = predict_upcoming_game(
    model=model,
    scaler=scaler,
    df_preprocessed=df_preprocessed,
    date="2024-03-05",
    home=1,           # Suppose the "team" is playing at home
    team= home_team,    # Must match the original categories
    team_opp= opp_team,  # Must match the original categories
    season=2024,
    fg_max=avg_fg_max_tor, fga_max=avg_fga_max_tor,   # Because we don't know them yet
    three_p_max=avg_3p_max_tor, three_pa_max=avg_3pa_max_tor,
    ft_max=avg_ft_max_tor, fta_max=avg_fta_max_tor,
    fg_max_opp=avg_fg_max_nop, fga_max_opp=avg_fga_max_nop,
    three_p_max_opp=avg_3p_max_nop, three_pa_max_opp=avg_3pa_max_nop,
    ft_max_opp=avg_ft_max_nop, fta_max_opp=avg_fta_max_nop
)

print("Predicted label:", pred_label)
print("Probability of winning:", pred_prob)
