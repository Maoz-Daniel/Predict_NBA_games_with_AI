import os
import re

# Path to the folder containing HTML files
SCORES_DIR = "data/scores"

# Function to uncomment the line_score table in the HTML file
def uncomment_line_score(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()

    # Regular expression to match comments around the div containing the line_score table
    uncommented_content = re.sub(
        r"<!--\s*\n(<div class=\"table_container\" id=\"div_line_score\">.*?</div>)\s*\n-->",
        r"\1",
        content,
        flags=re.DOTALL
    )

    # Only write the file if changes were made
    if content != uncommented_content:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(uncommented_content)
        print(f"Modified {filepath}")
    else:
        print(f"No changes needed for {filepath}")

# Loop through all HTML files in the scores directory
for filename in os.listdir(SCORES_DIR):
    if filename.endswith(".html"):
        filepath = os.path.join(SCORES_DIR, filename)
        print(f"Uncommenting line_score table in {filename}...")
        uncomment_line_score(filepath)

print("All files processed successfully.")
