#!/usr/bin/env python3
# reduce_cic_ids2018.py - Reduces the CIC-IDS2018 NetFlow dataset while preserving distribution

import pandas as pd
import numpy as np
import os
import argparse

def reduce_dataset(input_file, output_file, reduction_rate=0.1, seed=42):
    """
    Reduces the CIC-IDS2018 dataset while preserving attack distribution.
    
    Args:
        input_file: Path to the input dataset file
        output_file: Path to save the reduced dataset
        reduction_rate: Fraction of data to keep (default: 0.1 = 10%)
        seed: Random seed for reproducibility
    """
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset size: {len(df)} records")
    
    # Display attack distribution
    print("\nOriginal attack distribution:")
    attack_counts = df['Attack'].value_counts()
    for attack, count in attack_counts.items():
        print(f"  {attack}: {count} ({count/len(df):.2%})")
    
    # Group by attack type and sample from each group
    print("\nSampling by attack type...")
    
    # Create different sampling rates for different attack types
    # Ensure rare attacks are kept at higher rates
    attack_types = df['Attack'].unique()
    sample_rates = {}
    
    # For very rare attacks (less than 1% of dataset), keep a higher percentage
    threshold_rare = len(df) * 0.01
    threshold_uncommon = len(df) * 0.1
    
    for attack in attack_types:
        attack_count = len(df[df['Attack'] == attack])
        
        if attack_count < threshold_rare:
            # Very rare attacks - keep at least 50% or more
            sample_rates[attack] = max(0.5, reduction_rate * 5)
            print(f"  {attack}: Very rare attack ({attack_count} records) - keeping {sample_rates[attack]:.0%}")
        
        elif attack_count < threshold_uncommon:
            # Uncommon attacks - keep a higher percentage than base rate
            sample_rates[attack] = min(1.0, reduction_rate * 2)
            print(f"  {attack}: Uncommon attack ({attack_count} records) - keeping {sample_rates[attack]:.0%}")
        
        else:
            # Common attacks - use the base reduction rate
            # Adjust base rate downward to compensate for higher rates on rare classes
            sample_rates[attack] = reduction_rate * 0.9  
            print(f"  {attack}: Common attack ({attack_count} records) - keeping {sample_rates[attack]:.0%}")
    
    # Perform stratified sampling
    sampled_dfs = []
    
    for attack, rate in sample_rates.items():
        class_df = df[df['Attack'] == attack]
        
        # If the class is tiny, keep all samples
        if len(class_df) < 1000:
            sampled_dfs.append(class_df)
            print(f"Keeping all {len(class_df)} samples of attack type {attack}")
        else:
            # Otherwise sample according to our rates
            sampled_class = class_df.sample(frac=rate, random_state=seed)
            sampled_dfs.append(sampled_class)
            print(f"Sampled {len(sampled_class)} from {len(class_df)} for attack type {attack} ({rate:.0%})")
    
    # Combine all sampled classes
    reduced_df = pd.concat(sampled_dfs)
    
    # Shuffle the dataset
    reduced_df = reduced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Display new distribution
    print("\nReduced dataset distribution:")
    new_attack_counts = reduced_df['Attack'].value_counts()
    for attack, count in new_attack_counts.items():
        original_count = attack_counts[attack]
        print(f"  {attack}: {count} ({count/len(reduced_df):.2%}) - Reduction: {count/original_count:.2%}")
    
    print(f"\nFinal dataset size: {len(reduced_df)} records ({len(reduced_df)/len(df):.2%} of original)")
    
    # Save to output file
    print(f"Saving reduced dataset to {output_file}...")
    reduced_df.to_csv(output_file, index=False)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description='Reduce CIC-IDS2018 NetFlow dataset size while preserving distribution')
    parser.add_argument('--input', default='/media/ssd/test/standardized-datasets/netflow/nf_cic_ids2018_standardized.csv',
                        help='Path to input dataset file')
    parser.add_argument('--output', default='/media/ssd/test/standardized-datasets/netflow/nf_cic_ids2018_reduced.csv',
                        help='Path to save reduced dataset')
    parser.add_argument('--rate', type=float, default=0.1,
                        help='Reduction rate (fraction of data to keep, default: 0.1 = 10%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    reduce_dataset(args.input, args.output, args.rate, args.seed)

if __name__ == '__main__':
    main() 