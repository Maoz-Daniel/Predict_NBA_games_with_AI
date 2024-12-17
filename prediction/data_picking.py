from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
TOP_30 = ['mp', 'ft', 'fta', 'ft%', 'orb', 'drb', 'trb', 
          'drb%', 'ast%', 'tov%', 'usg%', 'orb_max', 'drb%_max',
            'blk%_max', 'usg%_max', 'mp_opp', 'fg%_opp', 'fta_opp', 
            'orb_opp', 'orb%_opp', 'stl%_opp', 'usg%_opp', 'fga_max_opp',
              '3p_max_opp', 'ft_max_opp', 'ft%_max_opp', 'orb_max_opp', 
              'drb_max_opp', '3par_max_opp', 'drb%_max_opp']

def save_data(df, output_path):
    """
    Save the cleaned dataset to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []

    seasons = sorted(data["season"].unique())

    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]

        model.fit(train[predictors], train["target"])

        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "predicted"]

        all_predictions.append(combined)
    return pd.concat(all_predictions)
    

def predictors(rr, split, sfs, df, selected_coulmns):
    
    scaler = MinMaxScaler() # scale the data to be between 0 and 1
    df[selected_coulmns] = scaler.fit_transform(df[selected_coulmns]) # scale the data

    sfs.fit(df[selected_coulmns], df["target"])
    predictors = list(selected_coulmns[sfs.get_support()]) # get the selected features
    return predictors

def find_team_averages(team):
    numeric_team = team.select_dtypes(include=["number"])  # Select only numeric columns
    rolling = numeric_team.rolling(10).mean()
    return rolling

    
def main():
    rr= RidgeClassifier(alpha=1.0)
    Split= TimeSeriesSplit(n_splits=3)
    sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction='forward', cv=Split)
    df = pd.read_csv("data\\games_all_clean.csv", index_col=None)

    removed_columns = ["season", "date", "won", "target", "Team", "Team_opp"] # remove columns that are not relevant for the prediction
    selected_coulmns = df.columns[~df.columns.isin(removed_columns)]

    #predictors = predictors(rr, split, sfs, df, selected_coulmns) # get the predictors
    predictors = TOP_30
    prediction = backtest(df, rr, predictors) # backtest the model
    prediction = prediction[prediction["actual"] != 2] # remove the games that were not played
    accuracy_score(prediction["actual"], prediction["predicted"])

    df_rolling = df[list(selected_coulmns) + ["won", "Team", "season"]]  # Keep relevant columns
    df_rolling = df_rolling.groupby(["Team", "season"], group_keys=False).apply(find_team_averages)  # Apply rolling mean
    print(df_rolling)
    #39:34




    #df.groupby("home").apply(lambda x: x[x["won"] == 1].shape[0] / x.shape[0]) # get the win rate for the home team)


    


# --------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- There might be a better way to select the features -----------------------------------------------
# -----------------------------25,21 time in yt---------------------------------------------------------------------------------------------------------------------


__name__ == "__main__"
main()
    