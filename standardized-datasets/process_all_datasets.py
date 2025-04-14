#!/usr/bin/env python3
# process_all_datasets.py - Runs the entire dataset processing pipeline

import os
import argparse
import subprocess
import time

def run_command(command, description):
    """Run a command and print its output"""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(command)}")
    start_time = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Stream the output as it comes
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    elapsed_time = time.time() - start_time
    
    if process.returncode == 0:
        print(f"\nCommand completed successfully in {elapsed_time:.2f} seconds")
        return True
    else:
        print(f"\nCommand failed with exit code {process.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Process all datasets: reduce, standardize, and combine')
    parser.add_argument('--bot-iot-v2-rate', type=float, default=0.01,
                        help='Reduction rate for BoT-IoT v2 dataset (default: 0.01)')
    parser.add_argument('--cic-ids-rate', type=float, default=0.1,
                        help='Reduction rate for CIC-IDS2018 dataset (default: 0.1)')
    parser.add_argument('--unsw-rate', type=float, default=0.1,
                        help='Reduction rate for UNSW-NB15 dataset (default: 0.1)')
    parser.add_argument('--skip-reduce', action='store_true',
                        help='Skip the dataset reduction step')
    parser.add_argument('--skip-standardize', action='store_true',
                        help='Skip the dataset standardization step')
    parser.add_argument('--skip-combine', action='store_true',
                        help='Skip the dataset combination step')
    parser.add_argument('--use-v2', action='store_true',
                        help='Use BoT-IoT v2 instead of v1')
    
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Reduce datasets
    if not args.skip_reduce:
        # Reduce BoT-IoT v2
        bot_iot_v2_cmd = [
            'python3', os.path.join(script_dir, 'reduce_bot_iot_v2.py'),
            '--rate', str(args.bot_iot_v2_rate)
        ]
        if not run_command(bot_iot_v2_cmd, "Reducing BoT-IoT v2 dataset"):
            print("Warning: BoT-IoT v2 reduction failed, but continuing with other steps")
        
        # Reduce CIC-IDS2018
        cic_ids_cmd = [
            'python3', os.path.join(script_dir, 'reduce_cic_ids2018.py'),
            '--rate', str(args.cic_ids_rate)
        ]
        if not run_command(cic_ids_cmd, "Reducing CIC-IDS2018 dataset"):
            print("Warning: CIC-IDS2018 reduction failed, but continuing with other steps")
        
        # Reduce UNSW-NB15
        unsw_cmd = [
            'python3', os.path.join(script_dir, 'reduce_unsw_nb15.py'),
            '--rate', str(args.unsw_rate)
        ]
        if not run_command(unsw_cmd, "Reducing UNSW-NB15 dataset"):
            print("Warning: UNSW-NB15 reduction failed, but continuing with other steps")
    
    # Step 2: Standardize datasets
    if not args.skip_standardize:
        std_cmd = ['python3', os.path.join(script_dir, 'standardize_datasets.py')]
        
        # Add use-v2 flag if specified
        if args.use_v2:
            std_cmd.append('--use-v2')
        
        if not run_command(std_cmd, "Standardizing datasets"):
            print("Error: Dataset standardization failed")
            return
    
    # Step 3: Combine datasets
    if not args.skip_combine:
        # Combine standardized datasets (full versions)
        combine_cmd = ['python3', os.path.join(script_dir, 'combine_datasets.py')]
        
        # Add use-v2 flag if specified
        if args.use_v2:
            combine_cmd.append('--use-v2')
        
        if not run_command(combine_cmd, "Combining standardized datasets (full versions)"):
            print("Warning: Failed to combine standard datasets")
        
        # Combine reduced standardized datasets
        combine_reduced_cmd = ['python3', os.path.join(script_dir, 'combine_datasets.py'), '--reduced']
        
        # Add use-v2 flag if specified
        if args.use_v2:
            combine_reduced_cmd.append('--use-v2')
        
        if not run_command(combine_reduced_cmd, "Combining standardized datasets (reduced versions)"):
            print("Warning: Failed to combine reduced datasets")
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED")
    print(f"{'='*80}")
    print("All datasets have been processed. You can find:")
    print("- Reduced datasets in: netflow/")
    print("- Standardized datasets in: netflow/")
    print("- Combined datasets in: combined/")

if __name__ == '__main__':
    main() 