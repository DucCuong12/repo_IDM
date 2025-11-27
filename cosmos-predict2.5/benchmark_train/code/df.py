import pandas as pd

data = []
for i in range(1, 101):
    file_name = f"{i}.txt"
    text = "Robot uses left hand to grip the edge of the bearing and put it into the box"
    data.append({"file_name": file_name, "text": text})

df = pd.DataFrame(data)
df.to_csv("metadata.csv", index=False)
print("metadata.csv created!")
