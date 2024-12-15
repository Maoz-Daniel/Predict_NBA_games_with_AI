import pandas as pd

def combine():
    games_clean = pd.read_csv("data/games.csv")
    new_clean = pd.read_csv("data/new.csv")

    # Step 1: Drop redundant index columns
    games_clean = games_clean.drop(columns=["Unnamed: 0"], errors='ignore')
    new_clean = new_clean.drop(columns=["Unnamed: 0.1"], errors='ignore')

    # Step 2: Align columns
    # Identify columns unique to each dataset
    games_clean_columns = set(games_clean.columns)
    new_clean_columns = set(new_clean.columns)

    missing_in_games_clean = new_clean_columns - games_clean_columns
    missing_in_new_clean = games_clean_columns - new_clean_columns

    print("Columns in new_clean but not in games_clean:", missing_in_games_clean)
    print("Columns in games_clean but not in new_clean:", missing_in_new_clean)

    # Optionally drop non-matching columns or add placeholders
    for col in missing_in_games_clean:
        games_clean[col] = None

    for col in missing_in_new_clean:
        new_clean[col] = None

    # Step 3: Combine datasets
    combined_dataset = pd.concat([games_clean, new_clean], ignore_index=True)

    # Save the cleaned and combined dataset for later use
    combined_dataset.to_csv("data\combined_uncleaned_dataset.csv", index=False)

    # Output the final shape and column structure
    combined_dataset.info()


def print_dataset(dataset):
    print(dataset)

def delete_column(dataset, column_name):
    dataset = dataset.drop(columns=[column_name], errors='ignore')
    dataset.to_csv("data\combined_uncleaned_dataset1.csv", index=False)
    return dataset


def main():
    # combine()
    unclean_combined_dataset = pd.read_csv("data\combined_uncleaned_dataset.csv")
    # remove unnamed column
    delete_column(unclean_combined_dataset, "Unnamed: 0")
    # combined_dataset = combined_dataset.drop(columns=["Unnamed: 0"], errors='ignore')
    print_dataset(unclean_combined_dataset)
    


if __name__ == "__main__":
    main()
