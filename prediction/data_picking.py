from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


rr= RidgeClassifier(alpha=1.0)
Split= TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(rr, n_features_to_select=30, direction='forward', cv=Split)
df = pd.read_csv("data\\games_clean.csv", index_col=None)
df = df.drop(columns=["Unnamed: 0"])

removed_columns = ["season", "date", "won", "target", "Team", "Team_opp"] # remove columns that are not relevant for the prediction
selected_coulmns = df.columns[~df.columns.isin(removed_columns)]


scaler = MinMaxScaler() # scale the data to be between 0 and 1
df[selected_coulmns] = scaler.fit_transform(df[selected_coulmns]) # scale the data

sfs.fit(df[selected_coulmns], df["target"])
preditros = list(selected_coulmns[sfs.get_support()]) # get the selected features
for i in preditros:
    print(i)

# --------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- There might be a better way to select the features -----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------


# def backtest(data, model, predictors, start=2, step=1):
    