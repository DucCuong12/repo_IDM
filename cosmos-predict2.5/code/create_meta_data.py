import csv

# Define output CSV file name
output_file = "metadata.csv"

# Open the file for writing
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["file_name", "text"])  # Header row

    # 1.mp4 -> 100.mp4
    for i in range(1, 101):
        writer.writerow([f"{i}.mp4", "Robot uses left hand to grip the edge of the bearing and put it into the box"])

    # 101.mp4 -> 203.mp4
    for i in range(101, 205):
        writer.writerow([f"{i}.mp4", "Robot's left-hand fingers hold the bottle securely and move it into the dish"])

print(f"✅ CSV file '{output_file}' created successfully!")

