import pandas as pd
import os

def combine_csv_files():
    """Combine two CSV files with identical columns into a single file."""
    try:
        # Paths to the files
        file1 = "data/games_all_clean.csv"
        file2 = "data/newGames_clean.csv"
        save_clean = "data/games_all_clean.csv"
    

        # Load both CSV files
        games_clean = pd.read_csv(file1)
        new_clean = pd.read_csv(file2)

        # Display loaded data for debugging

        # Append the two datasets
        if(new_clean is not None):
            combined_dataset = pd.concat([games_clean, new_clean], ignore_index=True)

        # Remove duplicates
        combined_dataset.drop_duplicates(inplace=True)

        # Sort the data by date and reset the index (if 'date' column exists)
        if 'date' in combined_dataset.columns:
            combined_dataset = combined_dataset.sort_values(by='date').reset_index(drop=True)
        else:
            print("Warning: 'date' column not found. Skipping sorting.")
        print(len(combined_dataset))
        # Save the combined dataset
        combined_dataset.to_csv(save_clean, index=False)
        print(f"Combined dataset saved to {save_clean}")



    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    combine_csv_files()
