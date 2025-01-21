from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_rolling_means(df, columns, window=10):
    """
    Calculate rolling means for specified columns over a given window.
    """
    for col in columns:
        df[f"{col}_rolling_{window}"] = df.groupby("Team")[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
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

def grid_search(data, predictors):
    """
    Perform grid search with cross-validation to find the best hyperparameters for the Random Forest model.
    """
    params = {
        'max_depth': [6, 8, None],
        'n_estimators': [50, 150, 250],
        'min_samples_split': [0.02, 0.05],
        'min_samples_leaf': [0.01, 0.05]
    }
    clf = RandomForestClassifier()
    grid = GridSearchCV(clf, params, cv=50, scoring='accuracy', verbose=2)
    grid.fit(data[predictors], data['won'])
    return grid.best_estimator_, grid.best_params_

def main():
    # Load and preprocess data
    df = pd.read_csv("data\games_all_clean.csv")

    removed_columns = ["season", "date", "target", "Team", "Team_opp"]
    selected_columns = df.columns[~df.columns.isin(removed_columns)]

    # Calculate rolling averages for key columns
    rolling_columns = [
           "fg%", "3p%", "ft%","trb", "ast", "stl", "blk", "tov", "pts",
        "ts%", 'ft%', 'ast_opp', 'ft%_opp', 'trb_opp', 
          '3p%_opp','fg%_opp'
    ]
    df = calculate_rolling_means(df, rolling_columns, window=10)

    # Calculate rolling win percentage
    df = calculate_win_percentage(df, window=10)

    # Use only the rolling averages and win percentage for modeling
    modeling_columns = [f"{col}_rolling_10" for col in rolling_columns] + ["win_percentage_rolling"]

    scaler = MinMaxScaler()
    df[modeling_columns] = scaler.fit_transform(df[modeling_columns])

    # Perform PCA to reduce dimensionality
    pca_components = 10  # Set number of components as needed
    pca = PCA(n_components=pca_components)
    reduced_features = pca.fit_transform(df[modeling_columns])
    reduced_df = pd.DataFrame(reduced_features, columns=[f"PC{i+1}" for i in range(pca_components)])

    # Add the target column back to the reduced data
    reduced_df['won'] = df['won']
    reduced_df['season'] = df['season']

    predictors = [col for col in reduced_df.columns if col not in ['won', 'season']]

    # Perform Grid Search
    print("Performing Grid Search...")
    best_model, best_params = grid_search(reduced_df, predictors)
    print(f"Best Parameters: {best_params}")

    # Evaluate the model
    print("Evaluating the model...")
    accuracy, f1, roc_auc = evaluate_model(reduced_df, best_model, predictors)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()
