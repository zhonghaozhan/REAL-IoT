#!/usr/bin/env python3
# combine_datasets.py - Combines standardized NetFlow datasets

import os
import pandas as pd
import argparse

def combine_netflow_datasets(input_dir, output_dir, use_reduced=False, use_bot_iot_v2=False):
    """
    Combine all standardized NetFlow datasets into one
    
    Args:
        input_dir: Directory containing standardized NetFlow datasets
        output_dir: Directory to save the combined dataset
        use_reduced: Whether to use reduced dataset versions if available
        use_bot_iot_v2: Whether to use BoT-IoT v2 instead of v1 (if both exist)
    
    Returns:
        Path to the combined dataset file
    """
    print("Combining NetFlow datasets...")
    
    # Define file patterns based on whether to use reduced datasets
    if use_reduced:
        # Make sure we're using standardized versions of reduced datasets
        file_pattern = "nf_{}_reduced_standardized.csv"
        # Try first standardized version of the reduced dataset
        fallback_pattern_1 = "nf_{}_reduced.csv"
        # Otherwise fall back to regular standardized version
        fallback_pattern_2 = "nf_{}_standardized.csv"
        output_file_name = "combined_netflow_reduced.csv"
    else:
        file_pattern = "nf_{}_standardized.csv"
        fallback_pattern_1 = None
        fallback_pattern_2 = None
        output_file_name = "combined_netflow.csv"
    
    # List of standardized NetFlow datasets
    if use_bot_iot_v2:
        # If using v2, don't include v1
        dataset_names = ['bot_iot_v2', 'cic_ids2018', 'unsw_nb15']
    else:
        # Default dataset list
        dataset_names = ['bot_iot', 'cic_ids2018', 'unsw_nb15']
    
    # Check if we can include the v2 dataset as well (when not using v2 exclusively)
    if not use_bot_iot_v2:
        v2_file = os.path.join(input_dir, file_pattern.format('bot_iot_v2'))
        v2_fallback_1 = None
        v2_fallback_2 = None
        
        if fallback_pattern_1:
            v2_fallback_1 = os.path.join(input_dir, fallback_pattern_1.format('bot_iot_v2'))
        if fallback_pattern_2:
            v2_fallback_2 = os.path.join(input_dir, fallback_pattern_2.format('bot_iot_v2'))
        
        # If v2 exists in any form, add it to the dataset list
        if os.path.exists(v2_file) or (v2_fallback_1 and os.path.exists(v2_fallback_1)) or (v2_fallback_2 and os.path.exists(v2_fallback_2)):
            print("BoT-IoT v2 dataset found, including it in combination")
            dataset_names.append('bot_iot_v2')
    
    dataset_files = []
    
    # Build the list of files to combine
    for dataset_name in dataset_names:
        primary_file = os.path.join(input_dir, file_pattern.format(dataset_name))
        
        if os.path.exists(primary_file):
            print(f"Using primary file for {dataset_name}: {primary_file}")
            dataset_files.append((dataset_name, primary_file))
        elif fallback_pattern_1:
            # Try the first fallback pattern
            fallback_file_1 = os.path.join(input_dir, fallback_pattern_1.format(dataset_name))
            if os.path.exists(fallback_file_1):
                print(f"Note: Primary file for {dataset_name} not found, using first fallback: {fallback_file_1}")
                dataset_files.append((dataset_name, fallback_file_1))
            elif fallback_pattern_2:
                # Try the second fallback pattern
                fallback_file_2 = os.path.join(input_dir, fallback_pattern_2.format(dataset_name))
                if os.path.exists(fallback_file_2):
                    print(f"Note: Reduced file for {dataset_name} not found, using standard version: {fallback_file_2}")
                    dataset_files.append((dataset_name, fallback_file_2))
                else:
                    print(f"Warning: No suitable dataset file for {dataset_name} found")
            else:
                print(f"Warning: No suitable dataset file for {dataset_name} found")
        else:
            print(f"Warning: Dataset file for {dataset_name} not found")
    
    # Load and combine datasets
    combined_df = pd.DataFrame()
    
    # Process and combine each dataset
    for dataset_name, dataset_file in dataset_files:
        print(f"Loading NetFlow {dataset_name} dataset from {dataset_file}")
        df = pd.read_csv(dataset_file)
        
        # Add dataset source column if not already present
        if 'dataset_source' not in df.columns:
            df['dataset_source'] = dataset_name
        
        # Combine with main dataframe
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    if not combined_df.empty:
        # Re-index flow_id to ensure uniqueness
        combined_df['flow_id'] = range(len(combined_df))
        
        # Save combined dataset
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_file_name)
        combined_df.to_csv(output_file, index=False)
        
        print(f"Combined NetFlow dataset saved to {output_file}")
        print(f"Total records: {len(combined_df)}")
        print(f"Records per dataset source:")
        for source in combined_df['dataset_source'].unique():
            count = len(combined_df[combined_df['dataset_source'] == source])
            print(f"  {source}: {count}")
        
        # Print attack distribution
        print(f"Attack type distribution:")
        attack_counts = combined_df['Attack'].value_counts()
        for attack, count in attack_counts.items():
            print(f"  {attack}: {count} ({count/len(combined_df):.2%})")
        
        return output_file
    else:
        print("No NetFlow datasets found to combine")
        return None

def main():
    parser = argparse.ArgumentParser(description='Combine standardized NetFlow datasets')
    parser.add_argument('--input_dir', default='netflow',
                      help='Input directory containing standardized NetFlow datasets')
    parser.add_argument('--output_dir', default='combined',
                      help='Output directory for combined datasets')
    parser.add_argument('--reduced', action='store_true',
                      help='Use reduced datasets if available')
    parser.add_argument('--use-v2', action='store_true',
                      help='Use BoT-IoT v2 instead of v1')
    
    args = parser.parse_args()
    
    # Resolve directories relative to standardized-datasets folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input_dir)
    output_dir = os.path.join(script_dir, args.output_dir)
    
    # Combine NetFlow datasets
    combine_netflow_datasets(input_dir, output_dir, args.reduced, args.use_v2)

if __name__ == '__main__':
    main() 