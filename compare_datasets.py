import pandas as pd

# Load both datasets
old_df = pd.read_csv("dataset/gestures.csv")
new_df = pd.read_csv("dataset/gestures_session2.csv")

# Feature columns
feature_cols = [col for col in old_df.columns if col != "label"]

print("===== DATASET COMPARISON =====")

print("\nOriginal dataset shape:", old_df.shape)
print("Session 2 dataset shape:", new_df.shape)

print("\nOriginal class counts:")
print(old_df["label"].value_counts())

print("\nSession 2 class counts:")
print(new_df["label"].value_counts())

# Compare mean coordinates for each gesture
print("\n===== MEAN COORDINATE COMPARISON =====")

for gesture in sorted(set(old_df["label"]) & set(new_df["label"])):
    old_mean = old_df.loc[
        old_df["label"] == gesture, feature_cols
    ].mean()

    new_mean = new_df.loc[
        new_df["label"] == gesture, feature_cols
    ].mean()

    mean_difference = (old_mean - new_mean).abs().mean()

    print(f"{gesture}: {mean_difference:.6f}")