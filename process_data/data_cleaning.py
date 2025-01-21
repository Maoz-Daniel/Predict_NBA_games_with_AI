import pandas as pd
import os
def load_and_clean_data(file_path):
    """
    Load the dataset and perform initial cleaning (sorting, resetting index, removing unnecessary columns).
    """
    df = pd.read_csv(file_path)
    df = df.sort_values(by='date').reset_index(drop=True)

    return df

def add_target_column(df):
    """
    Add the 'target' column to indicate the next game's result.
    """
    def add_target(team):
        team["target"] = team["won"].shift(-1)
        return team

    df = df.groupby("Team", group_keys=False).apply(add_target)

    # Set target to 2 for teams with no more games to play
    df["target"].fillna(2, inplace=True)

    # Convert target to integer
    df["target"] = df["target"].astype(int, errors="ignore")

    return df

def delete_irrelevant_columns(df, columns_to_delete):
    df.drop(columns=columns_to_delete, inplace=True)
    print("Successfully removed irrelevant columns.")

    return df

def fill_missing_values(df):
    """
    Fill missing values in the DataFrame using the average value for the field,
    calculated based on the team and season of the current row.
    """
    missing_summary = {}

    for index, row in df.iterrows():
        for column in df.columns:
            if pd.isna(row[column]):
                team = row['Team']
                season = row['season']
                
                relevant_rows = df[(df['Team'] == team) & (df['season'] == season)]
                average_value = relevant_rows[column].mean()

                df.at[index, column] = average_value

                if column not in missing_summary:
                    missing_summary[column] = 0
                missing_summary[column] += 1

    print("\n=== Summary of Missing Value Completion ===")
    for column, count in missing_summary.items():
        print(f"Column '{column}': {count} values filled.")

    return df

def save_clean_data(df, output_path):
    """
    Save the cleaned dataset to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

def check_for_nan_rows(df):
    """
    Check for rows with missing values and print their index and column details.
    """
    print("\nRows with missing values:")
    for index, row in df.iterrows():
        if row.isnull().any():
            nan_columns = row[row.isnull()].index.tolist()
            nan_values = row[row.isnull()].values.tolist()
            print(f"Row {index}: Columns with NaN - {nan_columns}, Values - {nan_values}")

def cheack_diffrent_columns(df1, df2):
    """
    print all the colums in df1 and not in df2, and all the columns in df2 and not in df1
    """
    columns1 = df1.columns
    columns2 = df2.columns

    diff1 = set(columns1) - set(columns2)
    diff2 = set(columns2) - set(columns1)

    print(f"Columns in df1 and not in df2: {diff1}")
    print(f"Columns in df2 and not in df1: {diff2}")

def combine_data(df1, df2):
    """
    Combine two DataFrames by concatenating them.
    """
    combined_df = pd.concat([df1, df2], ignore_index=True)
    #sort the data by date and reset the index
    combined_df = combined_df.sort_values(by='date').reset_index(drop=True)
    return combined_df

def add_target_2020(df):
    """
    Due to the Corona virus, the games haven't been played for 4 months,
      so the data mistakenly classifies the target of all teams as 2, 
      so we will change it manually.
    """
    

    return df

    
def main():
    input_path = "data\\newGames.csv"
    output_path = "data\\newGames_clean.csv"

    df = load_and_clean_data(input_path)
    df = add_target_column(df)
    columns_to_delete = ['gmsc', '+/-', 'mp_max', 'mp_max.1', 'gmsc_opp', '+/-_opp', 
                         'mp_max_opp', 'mp_max_opp.1', '+/-_max', '+/-_max_opp' ,'mp.1', 'mp_opp.1','index_opp','gmsc_max','gmsc_max_opp']
    df = delete_irrelevant_columns(df, columns_to_delete)
    df = fill_missing_values(df)
    check_for_nan_rows(df)





    save_clean_data(df, output_path)




    #---------------- data from 2017-2018 to 2019-2020 seasons----------------
"""
    input_path_old= "data\\games_2015-16--2021-22.csv"
    output_path_old= "data\\games_2015-16--2021-22_clean.csv"

    df_old_seasons= pd.read_csv("data\\games_2015-16--2021-22.csv")
    years = [2018, 2019, 2020]
    df_old_seasons = df_old_seasons[df_old_seasons['season'].isin(years)]

    #update the current file
    save_clean_data(df_old_seasons, input_path_old)

    df_old_seasons= load_and_clean_data(input_path_old)

    # change team column name to Team, from total to Total, from total_opp to Total_opp, from team_opp to Team_opp
    df_old_seasons.rename(columns={'team':'Team'}, inplace=True)
    df_old_seasons.rename(columns={'total':'Total'}, inplace=True)
    df_old_seasons.rename(columns={'total_opp':'Total_opp'}, inplace=True)
    df_old_seasons.rename(columns={'team_opp':'Team_opp'}, inplace=True)

    df_old_seasons = add_target_column(df_old_seasons)


    # delete irrelevant columns
    #add two culloms, gmsc and gmsc_opp with Nan values
    columns_to_delete = ['Unnamed: 0','+/-', 'mp_max', 'mp_max.1','+/-_opp', 
                         'mp_max_opp', 'mp_max_opp.1', '+/-_max', '+/-_max_opp']
    df_old_seasons = delete_irrelevant_columns(df_old_seasons, columns_to_delete)
   
    df_old_seasons = fill_missing_values(df_old_seasons)
    check_for_nan_rows(df_old_seasons)
    cheack_diffrent_columns(df, df_old_seasons)

    save_clean_data(df_old_seasons, output_path_old)

    #combine the two dataframes
    combined_df = combine_data(df, df_old_seasons)
    save_clean_data(combined_df, "data\\games_all_clean.csv")

    check_for_nan_rows(combined_df)
    print("now the data is clean and ready for analysis")
    print(combined_df)
    
    delete_irrelevant_columns(combined_df, ['target'])
    print("------------------------------------------------")
    print(combined_df.columns)
    add_target_column(combined_df)
    print("------------------------------------------------")
    print(combined_df.columns)

    print("Data cleaning completed.")
    print("number of target 2:")
    """

if __name__ == "__main__":
    main()