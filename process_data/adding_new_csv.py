import pandas as pd

#data cleaning
df = pd.read_csv("data\\new.csv", index_col=None)

df = df[df['date'] < '2020-12-22']
df = df.sort_values(by='date')
df = df.reset_index(drop=True)

del df['mp.1']
del df['mp_opp.1']
del df['index_opp']

def add_target(team): #the target is the next game's result
    team["target"] = team["won"].shift(-1) #shifts the target column up by 1
    return team

df=df.groupby("Team",group_keys=False).apply(add_target) #group by team and apply the add_target function



df["target"][pd.isnull(df["target"])]=2 #if the target is null, it means that the team has no more games to play, so we set the target to 2

df["target"]=df["target"].astype(int,errors="ignore") #convert the target to int

nulls=pd.isnull(df) #check for nulls
nulls=nulls.sum()

def delete_unrellevant_columns(df): #delete columns that are not relevant for the prediction
    del df['+/-']
    del df['mp_max']
    del df['mp_max.1']
    del df['+/-_opp']
    del df['mp_max_opp']
    del df['mp_max_opp.1']
    del df['+/-_max']
    del df['+/-_max_opp']
    print("-----------------------------------------------remove the unrellevant culloms succesfully-----------------------------------------------")

    return df


def fill_missing_values(df):
    """
    Fills missing values in the DataFrame using the average value for the field,
    calculated based on the team and season of the current row.
    Prints a summary of the changes made and the values filled.
    """
    missing_summary = {}  # Dictionary to track missing values filled per column
    filled_values = {}    # Dictionary to store the filled values for each column

    for index, row in df.iterrows():
        for column in df.columns:
            if pd.isna(row[column]):  # Check if the value is missing
                # Extract team and season for the current row
                team = row['Team']
                season = row['season']

                # Filter DataFrame to get relevant rows for the same team and season
                relevant_rows = df[(df['Team'] == team) & (df['season'] == season)]

                # Calculate the average value for the column (excluding NaN values)
                average_value = relevant_rows[column].mean()

                # Fill the missing value with the calculated average
                df.at[index, column] = average_value

                # Update summary
                if column not in missing_summary:
                    missing_summary[column] = 0
                    filled_values[column] = []
                missing_summary[column] += 1
                filled_values[column].append(average_value)

    # Print summary
    print("\n=== Summary of Missing Value Completion ===")
    total_filled = 0
    for column, count in missing_summary.items():
        print(f"Column '{column}': {count} values filled.")
        print(f"Values filled: {filled_values[column]}")
        total_filled += count
    print(f"Total values filled: {total_filled}")

    return df

df=delete_unrellevant_columns(df)

# print(nulls[nulls>0]) #print the columns with nulls
df=fill_missing_values(df)
nulls=pd.isnull(df) #check for nulls
nulls=nulls.sum()

#print(nulls[nulls>0]) #print the columns with nulls

df_clean = df.copy()
print(df_clean)

df_clean.to_csv("data\\new_clean.csv") #save the cleaned data to a new csv file