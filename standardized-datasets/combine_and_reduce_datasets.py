#!/usr/bin/env python3
# combine_and_reduce_datasets.py - Combines and reduces specified standardized NetFlow datasets

import os
import pandas as pd
import numpy as np
import argparse
import re # For extracting dataset names

def extract_dataset_name(filename):
    """Extracts a base dataset name from the filename."""
    basename = os.path.basename(filename)
    # Try to find known patterns
    match = re.search(r'nf_([a-zA-Z0-9_]+?)_(reduced_)?standardized\.csv', basename)
    if match:
        name_part = match.group(1)
        reduced_part = match.group(2) or ''
        # Specific replacements for clarity if needed
        if name_part == 'bot_iot_v2': name_part = 'botv2'
        elif name_part == 'cic_ids2018': name_part = 'cic18'
        elif name_part == 'unsw_nb15': name_part = 'unsw15'
        return f"{name_part}{'_red' if reduced_part else ''}"
    # Fallback if pattern doesn't match
    return os.path.splitext(basename)[0].replace('_standardized', '').replace('nf_', '')

def combine_and_reduce_datasets(input_files, output_file, reduction_rate=0.1, seed=42):
    """
    Combines specified standardized NetFlow datasets and reduces the result
    while preserving attack distribution.

    Args:
        input_files: List of paths to standardized NetFlow dataset CSV files.
        output_file: Path to save the combined and reduced dataset.
        reduction_rate: Fraction of data to keep (default: 0.1 = 10%)
        seed: Random seed for reproducibility

    Returns:
        Path to the combined reduced dataset file if successful, else None.
    """
    print("Combining and reducing specified NetFlow datasets...")
    print(f"Input files: {input_files}")
    print(f"Reduction rate: {reduction_rate:.2f}")
    print(f"Output file: {output_file}")

    # Load and combine datasets
    combined_df = pd.DataFrame()

    # Process and combine each dataset
    for dataset_file in input_files:
        if not os.path.exists(dataset_file):
            print(f"Warning: Input file not found: {dataset_file}. Skipping.")
            continue

        dataset_name = extract_dataset_name(dataset_file)
        print(f"Loading {dataset_name} dataset from {dataset_file}...")
        try:
            df = pd.read_csv(dataset_file, low_memory=False)
        except Exception as e:
            print(f"Error loading {dataset_file}: {e}. Skipping.")
            continue

        # Add dataset source column if not already present, or overwrite for consistency
        df['dataset_source'] = dataset_name

        # Combine with main dataframe
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    if combined_df.empty:
        print("No valid NetFlow datasets found to combine or loaded successfully.")
        return None

    # Re-index flow_id to ensure uniqueness across the combined set
    combined_df['flow_id'] = range(len(combined_df))

    print(f"\nCombined dataset size before reduction: {len(combined_df)} records")
    print(f"Records per dataset source:")
    source_counts_orig = combined_df['dataset_source'].value_counts()
    for source, count in source_counts_orig.items():
        print(f"  {source}: {count} ({count/len(combined_df):.2%})")

    # Print original attack distribution
    print(f"\nOriginal attack distribution:")
    attack_counts_orig = combined_df['Attack'].value_counts()
    for attack, count in attack_counts_orig.items():
        print(f"  {attack}: {count} ({count/len(combined_df):.2%})")

    # Create different sampling rates for different attack types
    attack_types = combined_df['Attack'].unique()
    sample_rates = {}

    threshold_rare = len(combined_df) * 0.001
    threshold_uncommon = len(combined_df) * 0.01

    print("\nCalculating sampling rates by attack type...")
    for attack in attack_types:
        attack_count = attack_counts_orig.get(attack, 0)

        # Define sampling rates based on rarity (ensure minimum sampling)
        base_sample_rate = max(0.01, reduction_rate) # Ensure at least 1% sampling

        if attack_count == 0:
             sample_rates[attack] = 0.0 # Should not happen if attack_types comes from df
        elif attack_count < threshold_rare:
            sample_rates[attack] = min(1.0, max(0.9, base_sample_rate * 10)) # Keep most very rare
            print(f"  {attack}: Very rare attack ({attack_count} records) - target rate {sample_rates[attack]:.1%}")
        elif attack_count < threshold_uncommon:
            sample_rates[attack] = min(0.8, max(0.1, base_sample_rate * 5)) # Keep higher % of uncommon
            print(f"  {attack}: Uncommon attack ({attack_count} records) - target rate {sample_rates[attack]:.1%}")
        elif attack_count < len(combined_df) * 0.05:
            sample_rates[attack] = min(0.6, max(0.05, base_sample_rate * 2.5))
            print(f"  {attack}: Moderate attack ({attack_count} records) - target rate {sample_rates[attack]:.1%}")
        else:
             # Adjust base rate downward slightly for common classes if reduction_rate is low
            adj_base_rate = base_sample_rate * 0.9 if reduction_rate <= 0.1 else base_sample_rate
            sample_rates[attack] = max(0.01, adj_base_rate)
            print(f"  {attack}: Common attack ({attack_count} records) - target rate {sample_rates[attack]:.1%}")

    # Also stratify by dataset source
    print("\nPerforming stratified sampling by attack type and dataset source...")

    combined_df['strata'] = combined_df['Attack'].astype(str) + '_' + combined_df['dataset_source'].astype(str)

    sampled_dfs = []
    grouped = combined_df.groupby('strata', group_keys=False)

    # Use apply function for potentially cleaner sampling per group
    def sample_group(group):
        attack = group['Attack'].iloc[0]
        rate = sample_rates.get(attack, reduction_rate) # Fallback to overall rate if needed
        n_samples = max(1, int(np.ceil(len(group) * rate))) # Ensure at least 1 sample if rate > 0
        if n_samples >= len(group):
             return group # Keep all if calculated sample size is >= group size
        return group.sample(n=n_samples, random_state=seed)

    reduced_df = grouped.apply(sample_group)

    # Drop the temporary strata column
    if 'strata' in reduced_df.columns:
        reduced_df = reduced_df.drop('strata', axis=1)

    # Shuffle the final dataset
    reduced_df = reduced_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Re-index flow_id again for the final reduced set
    reduced_df['flow_id'] = range(len(reduced_df))

    # Display new distribution by attack type
    print("\nReduced dataset attack distribution:")
    new_attack_counts = reduced_df['Attack'].value_counts()
    for attack, count in new_attack_counts.items():
        original_count = attack_counts_orig.get(attack, 0)
        original_pct_total = original_count / len(combined_df) if len(combined_df) > 0 else 0
        reduced_pct_total = count / len(reduced_df) if len(reduced_df) > 0 else 0
        reduction_factor = count / original_count if original_count > 0 else 0
        print(f"  {attack}: {count} ({reduced_pct_total:.2%}) - Orig: {original_count} ({original_pct_total:.2%}) - Kept: {reduction_factor:.1%}")

    # Display new distribution by dataset source
    print("\nReduced dataset source distribution:")
    new_source_counts = reduced_df['dataset_source'].value_counts()
    for source, count in new_source_counts.items():
        original_count = source_counts_orig.get(source, 0)
        original_pct_total = original_count / len(combined_df) if len(combined_df) > 0 else 0
        reduced_pct_total = count / len(reduced_df) if len(reduced_df) > 0 else 0
        reduction_factor = count / original_count if original_count > 0 else 0
        print(f"  {source}: {count} ({reduced_pct_total:.2%}) - Orig: {original_count} ({original_pct_total:.2%}) - Kept: {reduction_factor:.1%}")

    final_reduction_rate = len(reduced_df) / len(combined_df) if len(combined_df) > 0 else 0
    print(f"\nFinal dataset size: {len(reduced_df)} records ({final_reduction_rate:.2%} of original)")

    # Calculate and display memory savings
    original_size_mb = combined_df.memory_usage(deep=True).sum() / (1024 * 1024)
    reduced_size_mb = reduced_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Memory usage: {reduced_size_mb:.2f} MB (reduced from {original_size_mb:.2f} MB)")

    # Save to the specified output file
    output_dir = os.path.dirname(output_file)
    if output_dir: # Ensure output directory exists if specified in the path
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving reduced combined dataset to {output_file}...")
    reduced_df.to_csv(output_file, index=False)

    print("Done!")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Combine specified standardized NetFlow datasets and reduce the result.')
    parser.add_argument('--input_files', nargs='+', required=True,
                        help='List of paths to standardized NetFlow CSV files to combine.')
    parser.add_argument('--output_file', required=True,
                        help='Path for the output combined and reduced CSV file.')
    parser.add_argument('--rate', type=float, default=0.1,
                        help='Target reduction rate (fraction of data to keep, default: 0.1 = 10%%). Actual rate may vary due to stratification.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Combine and reduce NetFlow datasets
    combine_and_reduce_datasets(args.input_files, args.output_file, args.rate, args.seed)

if __name__ == '__main__':
    main() 