#!/usr/bin/env python3
# reduce_bot_iot_v2.py - Reduces the BoT-IoT v2 NetFlow dataset while preserving attack distribution
# 
# IMPORTANT: This script ONLY performs dataset reduction (Step 1 in the workflow).
# Standardization should be done as a separate step using standardize_datasets.py

import pandas as pd
import numpy as np
import os
import argparse

def reduce_dataset(input_file, output_file, reduction_rate=0.05, seed=42):
    """
    Reduces the BoT-IoT v2 dataset while preserving attack distribution.
    
    This script is part of a 3-step workflow:
    1. REDUCTION (this script): Reduce dataset size while preserving attack distribution
    2. STANDARDIZATION: Apply standardize_datasets.py to standardize format
    3. COMBINATION: Use combine_datasets.py to combine with other datasets
    
    Args:
        input_file: Path to the input dataset file
        output_file: Path to save the reduced dataset
        reduction_rate: Fraction of data to keep (default: 0.05 = 5%)
        seed: Random seed for reproducibility
    """
    print(f"Loading dataset from {input_file}...")
    
    # The Bot-IoT v2 dataset is very large (6GB), so we'll use chunks
    chunk_size = 1000000  # Process 1 million rows at a time
    
    # First, analyze the full dataset to get attack distribution
    print("Analyzing attack distribution (this may take a while)...")
    attack_counts = {}
    total_rows = 0
    
    # Read the file in chunks to count attack types
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        chunk_counts = chunk['Attack'].value_counts()
        
        # Update the total counts
        for attack, count in chunk_counts.items():
            if attack in attack_counts:
                attack_counts[attack] += count
            else:
                attack_counts[attack] = count
        
        total_rows += len(chunk)
        print(f"Processed {total_rows} rows so far...")
    
    print(f"Original dataset size: {total_rows} records")
    
    # Display attack distribution
    print("\nOriginal attack distribution:")
    for attack, count in attack_counts.items():
        print(f"  {attack}: {count} ({count/total_rows:.2%})")
    
    # Create different sampling rates for different attack types
    # Ensure rare attacks are kept at higher rates
    sample_rates = {}
    
    # For very rare attacks (less than 0.1% of dataset), keep a higher percentage
    threshold_rare = total_rows * 0.001
    threshold_uncommon = total_rows * 0.05
    
    for attack, count in attack_counts.items():
        if count < threshold_rare:
            # Very rare attacks - keep at least 50% or more
            sample_rates[attack] = max(0.5, reduction_rate * 10)
            print(f"  {attack}: Very rare attack ({count} records) - keeping {sample_rates[attack]:.0%}")
        
        elif count < threshold_uncommon:
            # Uncommon attacks - keep a higher percentage than base rate
            sample_rates[attack] = min(1.0, reduction_rate * 4)
            print(f"  {attack}: Uncommon attack ({count} records) - keeping {sample_rates[attack]:.0%}")
        
        else:
            # Common attacks - use the base reduction rate
            # Adjust base rate downward to compensate for higher rates on rare classes
            sample_rates[attack] = reduction_rate * 0.9  
            print(f"  {attack}: Common attack ({count} records) - keeping {sample_rates[attack]:.0%}")
    
    # Perform stratified sampling in chunks
    print("\nSampling by attack type...")
    
    # Create the output file and write the header
    first_chunk = True
    sampled_counts = {attack: 0 for attack in attack_counts.keys()}
    
    # Process the file in chunks for sampling
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        sampled_dfs = []
        
        # Process each attack type in the chunk
        for attack, rate in sample_rates.items():
            class_df = chunk[chunk['Attack'] == attack]
            
            if len(class_df) > 0:
                # If the class is tiny, keep all samples
                if attack_counts[attack] < 1000:
                    sampled_dfs.append(class_df)
                    sampled_counts[attack] += len(class_df)
                else:
                    # Otherwise sample according to our rates
                    sampled_class = class_df.sample(frac=rate, random_state=seed)
                    sampled_dfs.append(sampled_class)
                    sampled_counts[attack] += len(sampled_class)
        
        # Combine all sampled classes in this chunk
        if sampled_dfs:
            reduced_chunk = pd.concat(sampled_dfs)
            
            # Shuffle the chunk
            reduced_chunk = reduced_chunk.sample(frac=1, random_state=seed)
            
            # Write to file (append mode after first chunk)
            reduced_chunk.to_csv(output_file, mode='w' if first_chunk else 'a', 
                              header=first_chunk, index=False)
            first_chunk = False
            
            print(f"Processed chunk with {len(chunk)} rows, saved {len(reduced_chunk)} rows")
    
    # Display new distribution
    print("\nReduced dataset distribution:")
    total_sampled = sum(sampled_counts.values())
    for attack, count in sampled_counts.items():
        original_count = attack_counts[attack]
        print(f"  {attack}: {count} ({count/total_sampled:.2%}) - Reduction: {count/original_count:.2%}")
    
    print(f"\nFinal dataset size: {total_sampled} records ({total_sampled/total_rows:.2%} of original)")
    print("\nNEXT STEPS:")
    print("1. Use standardize_datasets.py to standardize this reduced dataset")
    print("2. Use combine_datasets.py to combine with other datasets")
    print("\nDone!")

def main():
    parser = argparse.ArgumentParser(description='Reduce BoT-IoT v2 NetFlow dataset size while preserving distribution')
    parser.add_argument('--input', default='/media/ssd/test/GNN/kaggle/input/NF-BoT-IoT/NF-BoT-IoT-v2.csv',
                        help='Path to input dataset file')
    parser.add_argument('--output', default='/media/ssd/test/standardized-datasets/netflow/nf_bot_iot_v2_reduced.csv',
                        help='Path to save reduced dataset')
    parser.add_argument('--rate', type=float, default=0.05,
                        help='Reduction rate (fraction of data to keep, default: 0.05 = 5%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    reduce_dataset(args.input, args.output, args.rate, args.seed)

if __name__ == '__main__':
    main() 