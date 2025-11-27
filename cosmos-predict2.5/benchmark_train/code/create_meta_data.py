import pandas as pd

# Path to your existing CSV file
csv_path = "metadata.csv"

# Load the existing CSV
df = pd.read_csv(csv_path)

# Create new rows for 101.txt → 204.txt
new_data = pd.DataFrame({
    "file_name": [f"{i}.txt" for i in range(101, 205)],
    "text": ["Robot's left-hand fingers hold the bottle securely and move it into the box"] * 104
})

# Append new rows to the existing DataFrame
df = pd.concat([df, new_data], ignore_index=True)

# Save back to CSV
df.to_csv(csv_path, index=False)

print("✅ Appended 101–204 successfully to metadata.csv")
