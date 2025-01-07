import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from datetime import timedelta

###########################
# GLOBAL MAPPING
###########################
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

###########################
# SAVING AND LOADING
###########################
def save_data(df, output_path):
    """
    Save the cleaned dataset to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

###########################
# ADD TEAM STATS
###########################
def add_team_stats(df):
    """
    Add historical statistics for each team before each row.
    """
    df = df.sort_values('date')  # Sort by date
    
    teams = df['Team'].unique()
    
    # Track for each team
    team_stats = {
        team: {
            'last_5_wins': 0,
            'season_wins': 0,
            'season_games': 0
        } 
        for team in teams
    }
    
    new_rows = []
    previous_season = None

    for idx, row in df.iterrows():
        current_season = row['season']
        team = row['Team']
        team_opp = row['Team_opp']
        
        # reset stats if new season
        if previous_season is not None and current_season != previous_season:
            team_stats = {
                t: {'last_5_wins': 0, 'season_wins': 0, 'season_games': 0} 
                for t in teams
            }
        previous_season = current_season

        # stats BEFORE the game
        new_row = row.copy()
        new_row['win_rate_5'] = (team_stats[team]['last_5_wins'] / 5
                                 if team_stats[team]['season_games'] >= 5 
                                 else 0.5)
        new_row['season_win_rate'] = (team_stats[team]['season_wins'] / team_stats[team]['season_games']
                                      if team_stats[team]['season_games'] > 0 
                                      else 0.5)
        
        new_row['opp_win_rate_5'] = (team_stats[team_opp]['last_5_wins'] / 5
                                     if team_stats[team_opp]['season_games'] >= 5 
                                     else 0.5)
        new_row['opp_season_win_rate'] = (
            team_stats[team_opp]['season_wins'] / team_stats[team_opp]['season_games']
            if team_stats[team_opp]['season_games'] > 0 else 0.5
        )
        
        new_rows.append(new_row)
        
        # update statistics AFTER the game
        won = row['won']
        team_stats[team]['season_games'] += 1
        team_stats[team]['season_wins']  += won
        if team_stats[team]['season_games'] >= 5:
            # naive approach for "last_5_wins"
            old_rolling = team_stats[team]['last_5_wins']
            new_rolling = (old_rolling * 5 + won - 0) / 5
            team_stats[team]['last_5_wins'] = new_rolling
        else:
            team_stats[team]['last_5_wins'] = won

    return pd.DataFrame(new_rows)

###########################
# PREPROCESSING
###########################
def preprocess_data(df_raw):
    """
    מבצעת על DataFrame גולמי (כמו בקובץ CSV) את כל שלבי ה-preprocessing:
    1) המרת date
    2) add_team_stats
    3) days_rest ו-opp_days_rest
    4) המרת teams למספרים
    5) החזרת העמודות הסופיות
    """
    # make a copy so as not to modify original
    df = df_raw.copy()

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Ensure sorting (usually add_team_stats sorts by date, but let's do it again after anyway)
    df = df.sort_values('date').reset_index(drop=True)

    # add stats
    df = add_team_stats(df)

    # add days rest
    df['days_rest'] = df.groupby('Team')['date'].diff().dt.days.fillna(3)
    df['opp_days_rest'] = df.groupby('Team_opp')['date'].diff().dt.days.fillna(3)

    # convert teams to numeric
    df['Team'] = df['Team'].map(teams_to_num)
    df['Team_opp'] = df['Team_opp'].map(teams_to_num)

    # final list of columns
    features = [
        'home', 'Team', 'Team_opp',
        'fg_max', 'fga_max', '3p_max', '3pa_max', 'ft_max', 'fta_max',
        'fg_max_opp', 'fga_max_opp', '3p_max_opp', '3pa_max_opp',
        'ft_max_opp', 'fta_max_opp',
        'win_rate_5', 'season_win_rate',
        'opp_win_rate_5', 'opp_season_win_rate',
        'days_rest', 'opp_days_rest'
    ]

    # שמים את ה־features + את הטארגט+date+season
    df = df[features + ['won', 'season', 'date']].reset_index(drop=True)
    return df

def load_and_preprocess(csv_path):
    """
    פונקציה קצרה שטוענת את ה-csv ומחזירה את ה-df שעבר preprocessing
    """
    df_raw = pd.read_csv(csv_path)
    df_pre = preprocess_data(df_raw)
    return df_pre

###########################
# TRAIN & EVALUATE
###########################
def train_and_evaluate(file_path):
    df = load_and_preprocess(file_path)

    # separate train vs test by season
    max_season = df['season'].max()
    train_mask = df['season'] < max_season

    X_train = df[train_mask].drop(['won','season','date'], axis=1)
    y_train = df[train_mask]['won']
    X_test  = df[~train_mask].drop(['won','season','date'], axis=1)
    y_test  = df[~train_mask]['won']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

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
    preds = model.predict(X_test_scaled)

    print(f"Model Accuracy: {accuracy_score(y_test, preds):.3f}")
    print("\nPerformance Report:")
    print(classification_report(y_test, preds))

    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    return model, scaler, df

###########################
# פונקציית עזר: יצירת שורת-משחק "גולמית"
###########################
def create_raw_game_row(
    date, 
    home, 
    team_str, 
    team_opp_str,
    season,
    fg_max, fga_max,
    three_p_max, three_pa_max,
    ft_max, fta_max,
    fg_max_opp, fga_max_opp,
    three_p_max_opp, three_pa_max_opp,
    ft_max_opp, fta_max_opp
):
    """
    מחזירה מילון (dict) בפורמט כמו בקובץ CSV המקורי,
    עם העמודות: date, home, Team, Team_opp, fg_max... וכו'
    אבל **Team ו-Team_opp נשארים כמחרוזות** (כי רק אחר כך נמיר עם map).
    """
    return {
        'date': pd.to_datetime(date),
        'home': home,
        'Team': team_str,   # נשאר string
        'Team_opp': team_opp_str,  # נשאר string
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
        'won': 0  # placeholder, לצורך אחידות
    }

###########################
# PREDICT UPCOMING GAME (WITH FULL PIPELINE)
###########################
def predict_upcoming_game_full(
    model,
    scaler,
    df_preprocessed,  # זה ה-DF שכבר מאומן
    df_raw_csv_path,  # כדי שנוכל "לקרוא מחדש" ולהוסיף למשחקים הגולמיים
    date,
    home,
    team_str,
    team_opp_str,
    season,
    fg_max=0, fga_max=0, 
    three_p_max=0, three_pa_max=0,
    ft_max=0, fta_max=0,
    fg_max_opp=0, fga_max_opp=0,
    three_p_max_opp=0, three_pa_max_opp=0,
    ft_max_opp=0, fta_max_opp=0
):
    """
    בגישה הזו, אנו:
    1) טוענים את ה-CSV הגולמי
    2) מוסיפים אליו את השורה החדשה (בפורמט גולמי, עם Team כמחרוזת)
    3) מריצים על הכל את preprocess_data כדי שיחושבו win_rate_5 וכו' מחדש
    4) מבודדים את השורה האחרונה (שאנו הוספנו)
    5) מנרמלים לפי ה-scaler הקיים ומנבאים עם model
    """

    # 1) טוענים את ה-dataframe הגולמי
    df_raw = pd.read_csv(df_raw_csv_path)

    # 2) בונים את השורה החדשה בפורמט גולמי (כמו ה-CSV)
    new_row_dict = create_raw_game_row(
        date=date,
        home=home,
        team_str=team_str,
        team_opp_str=team_opp_str,
        season=season,
        fg_max=fg_max,
        fga_max=fga_max,
        three_p_max=three_p_max,
        three_pa_max=three_pa_max,
        ft_max=ft_max,
        fta_max=fta_max,
        fg_max_opp=fg_max_opp,
        fga_max_opp=fga_max_opp,
        three_p_max_opp=three_p_max_opp,
        three_pa_max_opp=three_pa_max_opp,
        ft_max_opp=ft_max_opp,
        fta_max_opp=fta_max_opp
    )

    # נוודא שהתאריך באמת אחרון (או אחרי כולם)
    # אם לא - נדחוף אותו קצת קדימה, כדי שנקבל days_rest תקין
    max_date_in_csv = pd.to_datetime(df_raw['date']).max()
    new_date = pd.to_datetime(new_row_dict['date'])
    if new_date <= max_date_in_csv:
        new_date = max_date_in_csv + pd.Timedelta(days=1)
    new_row_dict['date'] = new_date

    # מוסיפים
    df_raw = pd.concat([df_raw, pd.DataFrame([new_row_dict])],
                       ignore_index=True)

    # 3) מעבדים מחדש: add_team_stats וכו'
    df_processed = preprocess_data(df_raw)

    # 4) מבודדים את השורה החדשה מה-df המעובד
    #    נזהה אותה לפי התאריך (שהוא האחרון)
    #    וגם לפי season, home, וכו' (ליתר בטחון)
    mask_new = (
        (df_processed['date'] == new_date) &
        (df_processed['Team'] == teams_to_num[team_str]) &
        (df_processed['Team_opp'] == teams_to_num[team_opp_str]) &
        (df_processed['season'] == season) &
        (df_processed['home'] == home)
    )
    row_new = df_processed[mask_new].copy()  # DataFrame של השורה החדשה

    if len(row_new) == 0:
        raise ValueError("לא נמצאה השורה החדשה לאחר העיבוד! בדוק את הנתונים.")

    # מורידים עמודות שלא נדרש להזין למודל
    final_features = [
        'home', 'Team', 'Team_opp',
        'fg_max', 'fga_max', '3p_max', '3pa_max', 'ft_max', 'fta_max',
        'fg_max_opp', 'fga_max_opp', '3p_max_opp', '3pa_max_opp',
        'ft_max_opp', 'fta_max_opp',
        'win_rate_5', 'season_win_rate',
        'opp_win_rate_5', 'opp_season_win_rate',
        'days_rest', 'opp_days_rest'
    ]

    # ניקח רק שורה אחת (אמור להיות 1)
    row_for_model = row_new.iloc[0][final_features].values.reshape(1, -1)

    # 5) מנרמלים ומנבאים
    row_scaled = scaler.transform(row_for_model)
    pred_label = model.predict(row_scaled)[0]
    pred_prob  = model.predict_proba(row_scaled)[0][1]

    return pred_label, pred_prob


###########################
# EXAMPLE MAIN
###########################
if __name__ == "__main__":
    file_path = "data/first_half_2024.csv"
    model, scaler, df_pre = train_and_evaluate(file_path)

    # נשמור את ה־df המעובד
    save_data(df_pre, "data/temp.csv")

    # ננבא משחק דמה בין "MIA" ל-"DET"
    # נשים סטט' מוזרות כדי לראות אם משפיע
    pred_label, pred_prob = predict_upcoming_game_full(
        model=model,
        scaler=scaler,
        df_preprocessed=df_pre,
        df_raw_csv_path=file_path,
        date="2024-03-06",   # התאריך (נדחף ליום מעבר למקסימום אם לא האחרון)
        home=1,
        team_str="POR",
        team_opp_str="OKC",
        season=2024,
        fg_max=999, fga_max=1500,
        three_p_max=500, three_pa_max=700,
        ft_max=300, fta_max=400,
        fg_max_opp=200, fga_max_opp=250,
        three_p_max_opp=10, three_pa_max_opp=15,
        ft_max_opp=20, fta_max_opp=25
    )
    print(f"Prediction: {pred_label}, Probability of winning: {pred_prob:.4f}")
