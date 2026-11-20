import pandas as pd

file_path = "dataset/gestures_session2.csv"

df = pd.read_csv(file_path)

df["label"] = df["label"].replace({
    "THANK YOU": "THANK_YOU"
})

df.to_csv(file_path, index=False)

print("Labels fixed successfully!")
print("\nUpdated sample counts:")
print(df["label"].value_counts())