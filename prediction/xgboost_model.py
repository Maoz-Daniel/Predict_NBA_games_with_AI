import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from datetime import timedelta

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
    
    # Convert categorical teams to numeric codes
    df['Team'] = df['Team'].astype('category').cat.codes
    df['Team_opp'] = df['Team_opp'].astype('category').cat.codes
    
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

# Run the enhanced model
file_path = "data\\games_all_clean.csv"
train_and_evaluate(file_path)
