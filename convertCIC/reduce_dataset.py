#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Read the full dataset
print("Reading dataset...")
df = pd.read_csv('/media/ssd/test/GNN/kaggle/input/BoT-IoT/bot.csv')  # adjust separator if needed

# Detect if we need a custom separator
if len(df.columns) == 1:
    print("Dataset appears to use non-comma separator, trying semicolon...")
    df = pd.read_csv('/media/ssd/test/GNN/kaggle/input/BoT-IoT/bot.csv', sep=';')

# Rename category to label if needed
if 'category' in df.columns and 'label' not in df.columns:
    df.rename(columns={"category": "label"}, inplace=True)

# Display original distribution
print("\nOriginal distribution:")
print(df.label.value_counts())
print(f"Total records: {len(df)}")

# Create sampling strategy - different sampling rates for different classes
# Keep 100% of rare classes, reduce common classes
sampling_rates = {
    'DDoS': 0.35,           # Take 35% of DDoS
    'DoS': 0.35,            # Take 35% of DoS
    'Reconnaissance': 0.70,  # Take 70% of Reconnaissance
    'Normal': 1.0,          # Keep all Normal
    'Theft': 1.0            # Keep all Theft
}

# Perform stratified sampling
print("\nSampling from each class...")
sampled_dfs = []
for label, rate in sampling_rates.items():
    class_df = df[df.label == label]
    # If class is tiny, keep all samples
    if len(class_df) < 1000:
        sampled_dfs.append(class_df)
        print(f"Keeping all {len(class_df)} samples of class {label}")
    else:
        # Otherwise sample according to our rates
        sampled_class = class_df.sample(frac=rate, random_state=42)
        sampled_dfs.append(sampled_class)
        print(f"Sampled {len(sampled_class)} from {len(class_df)} for class {label} ({rate:.0%})")

# Combine all sampled classes
reduced_df = pd.concat(sampled_dfs)

# Shuffle the dataset
reduced_df = reduced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display new distribution
print("\nReduced distribution:")
print(reduced_df.label.value_counts())
print(f"Total records: {len(reduced_df)}")
print(f"Reduction: {len(reduced_df)/len(df):.2%}")

# Save to new file
output_path = '/media/ssd/test/GNN/kaggle/input/BoT-IoT/bot_reduced.csv'
print(f"\nSaving to {output_path}...")
reduced_df.to_csv(output_path, index=False)
print("Done!") 