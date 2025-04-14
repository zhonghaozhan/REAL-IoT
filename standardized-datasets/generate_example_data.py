#!/usr/bin/env python3
# generate_example_data.py - Generate example data for testing

import os
import pandas as pd
import numpy as np
import random
import argparse
from faker import Faker

def generate_bot_iot_sample(output_dir, num_samples=1000):
    """
    Generate a sample of BoT-IoT dataset
    
    Args:
        output_dir: Directory to save the sample dataset
        num_samples: Number of samples to generate
    
    Returns:
        Path to the sample dataset file
    """
    print(f"Generating {num_samples} samples of BoT-IoT dataset")
    
    fake = Faker()
    
    # Create dataframe
    df = pd.DataFrame()
    
    # Generate IPs
    iot_ips = [fake.ipv4() for _ in range(50)]
    attack_ips = [fake.ipv4() for _ in range(20)]
    normal_ips = [fake.ipv4() for _ in range(30)]
    
    # Generate data
    src_ips = []
    dst_ips = []
    labels = []
    attacks = []
    
    for _ in range(num_samples):
        is_attack = random.random() < 0.7  # 70% attacks
        
        if is_attack:
            src_ip = random.choice(attack_ips)
            dst_ip = random.choice(iot_ips)
            label = 'Attack'
            attack = random.choice(['DDoS', 'DoS', 'Reconnaissance', 'Theft'])
        else:
            src_ip = random.choice(normal_ips)
            dst_ip = random.choice(iot_ips)
            label = 'Normal'
            attack = 'Normal'
        
        src_ips.append(src_ip)
        dst_ips.append(dst_ip)
        labels.append(label)
        attacks.append(attack)
    
    df['src_ip'] = src_ips
    df['src_port'] = np.random.randint(1024, 65535, num_samples)
    df['dst_ip'] = dst_ips
    df['dst_port'] = np.random.randint(1, 1024, num_samples)
    df['proto'] = np.random.choice(['tcp', 'udp', 'icmp'], num_samples)
    df['dur'] = np.random.uniform(0, 10, num_samples)
    df['bytes_in'] = np.random.randint(100, 10000, num_samples)
    df['bytes_out'] = np.random.randint(100, 10000, num_samples)
    df['pkts_in'] = np.random.randint(1, 100, num_samples)
    df['pkts_out'] = np.random.randint(1, 100, num_samples)
    df['label'] = labels
    df['attack'] = attacks
    
    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'bot_iot.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Sample BoT-IoT dataset saved to {output_file}")
    return output_file

def generate_cic_ids2018_sample(output_dir, num_samples=1000):
    """
    Generate a sample of CSE-CIC-IDS2018 dataset
    
    Args:
        output_dir: Directory to save the sample dataset
        num_samples: Number of samples to generate
    
    Returns:
        Path to the sample dataset file
    """
    print(f"Generating {num_samples} samples of CSE-CIC-IDS2018 dataset")
    
    fake = Faker()
    
    # Create dataframe
    df = pd.DataFrame()
    
    # Generate IPs
    src_ips = [fake.ipv4() for _ in range(num_samples)]
    dst_ips = [fake.ipv4() for _ in range(num_samples)]
    
    # Attack types
    attack_types = ['Benign', 'SSH-Bruteforce', 'FTP-BruteForce', 'DoS-GoldenEye', 
                   'DoS-Slowloris', 'DoS-Slowhttptest', 'DoS-Hulk', 'WebAttack']
    
    # Generate labels
    labels = np.random.choice(attack_types, num_samples, p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02])
    
    df['Source IP'] = src_ips
    df['Source Port'] = np.random.randint(1024, 65535, num_samples)
    df['Destination IP'] = dst_ips
    df['Destination Port'] = np.random.randint(1, 1024, num_samples)
    df['Protocol'] = np.random.randint(0, 17, num_samples)
    df['Protocol Name'] = np.random.choice(['TCP', 'UDP', 'ICMP'], num_samples)
    df['Flow Duration'] = np.random.uniform(0, 10000, num_samples)
    df['Total Fwd Packets'] = np.random.randint(1, 1000, num_samples)
    df['Total Backward Packets'] = np.random.randint(1, 1000, num_samples)
    df['Total Length of Fwd Packets'] = np.random.randint(100, 10000, num_samples)
    df['Total Length of Bwd Packets'] = np.random.randint(100, 10000, num_samples)
    df['FIN Flag Count'] = np.random.randint(0, 10, num_samples)
    df['Label'] = labels
    
    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'cic_ids2018.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Sample CSE-CIC-IDS2018 dataset saved to {output_file}")
    return output_file

def generate_unsw_nb15_sample(output_dir, num_samples=1000):
    """
    Generate a sample of UNSW-NB15 dataset
    
    Args:
        output_dir: Directory to save the sample dataset
        num_samples: Number of samples to generate
    
    Returns:
        Path to the sample dataset file
    """
    print(f"Generating {num_samples} samples of UNSW-NB15 dataset")
    
    fake = Faker()
    
    # Create dataframe
    df = pd.DataFrame()
    
    # Generate IPs
    src_ips = [fake.ipv4() for _ in range(num_samples)]
    dst_ips = [fake.ipv4() for _ in range(num_samples)]
    
    # Attack categories
    attack_cats = ['-', 'Fuzzers', 'Analysis', 'Backdoor', 'DoS', 'Exploits', 
                  'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
    
    # Generate labels and attack categories
    labels = np.random.binomial(1, 0.3, num_samples)  # 30% attacks
    attack_categories = []
    
    for label in labels:
        if label == 0:
            attack_categories.append('-')
        else:
            attack_categories.append(np.random.choice(attack_cats[1:]))
    
    df['srcip'] = src_ips
    df['sport'] = np.random.randint(1024, 65535, num_samples)
    df['dstip'] = dst_ips
    df['dsport'] = np.random.randint(1, 1024, num_samples)
    df['proto'] = np.random.choice(['tcp', 'udp', 'icmp'], num_samples)
    df['dur'] = np.random.uniform(0, 10, num_samples)
    df['sbytes'] = np.random.randint(100, 10000, num_samples)
    df['dbytes'] = np.random.randint(100, 10000, num_samples)
    df['spkts'] = np.random.randint(1, 100, num_samples)
    df['dpkts'] = np.random.randint(1, 100, num_samples)
    df['sttl'] = np.random.randint(1, 255, num_samples)
    df['dttl'] = np.random.randint(1, 255, num_samples)
    df['label'] = labels
    df['attack_cat'] = attack_categories
    
    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'unsw_nb15.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Sample UNSW-NB15 dataset saved to {output_file}")
    return output_file

def generate_netflow_sample(output_dir, dataset_name, num_samples=1000):
    """
    Generate a sample of NetFlow dataset
    
    Args:
        output_dir: Directory to save the sample dataset
        dataset_name: Name of the dataset (bot_iot, cic_ids2018, or unsw_nb15)
        num_samples: Number of samples to generate
    
    Returns:
        Path to the sample dataset file
    """
    print(f"Generating {num_samples} samples of NetFlow {dataset_name} dataset")
    
    fake = Faker()
    
    # Create dataframe
    df = pd.DataFrame()
    
    # Generate IPs
    src_ips = [fake.ipv4() for _ in range(num_samples)]
    dst_ips = [fake.ipv4() for _ in range(num_samples)]
    
    # Generate common fields
    df['src_ip'] = src_ips
    df['src_port'] = np.random.randint(1024, 65535, num_samples)
    df['dst_ip'] = dst_ips
    df['dst_port'] = np.random.randint(1, 1024, num_samples)
    df['protocol'] = np.random.randint(0, 17, num_samples)
    df['protocol_name'] = np.random.choice(['TCP', 'UDP', 'ICMP'], num_samples)
    df['duration'] = np.random.uniform(0, 10, num_samples)
    df['bytes_in'] = np.random.randint(100, 10000, num_samples)
    df['bytes_out'] = np.random.randint(100, 10000, num_samples)
    df['packets_in'] = np.random.randint(1, 100, num_samples)
    df['packets_out'] = np.random.randint(1, 100, num_samples)
    
    # Generate dataset-specific fields
    if dataset_name == 'bot_iot':
        # 70% attacks
        labels = np.random.binomial(1, 0.7, num_samples)
        df['binary_label'] = labels
        
        # Attack types
        attack_types = ['Normal', 'DDoS', 'DoS', 'Reconnaissance', 'Theft']
        attacks = []
        
        for label in labels:
            if label == 0:
                attacks.append('Normal')
            else:
                attacks.append(np.random.choice(attack_types[1:]))
        
        df['attack_type'] = attacks
        
    elif dataset_name == 'cic_ids2018':
        # Attack types
        attack_types = ['Normal', 'SSH-Bruteforce', 'FTP-BruteForce', 'DoS-GoldenEye', 
                       'DoS-Slowloris', 'DoS-Slowhttptest', 'DoS-Hulk', 'WebAttack']
        
        # Generate attacks (30% attacks)
        attacks = np.random.choice(attack_types, num_samples, 
                                 p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02])
        
        # Binary labels
        labels = np.zeros(num_samples)
        labels[attacks != 'Normal'] = 1
        
        df['binary_label'] = labels
        df['attack_type'] = attacks
        
    elif dataset_name == 'unsw_nb15':
        # Attack categories
        attack_cats = ['Normal', 'Fuzzers', 'Analysis', 'Backdoor', 'DoS', 'Exploits', 
                      'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
        
        # Generate labels (30% attacks)
        labels = np.random.binomial(1, 0.3, num_samples)
        attack_categories = []
        
        for label in labels:
            if label == 0:
                attack_categories.append('Normal')
            else:
                attack_categories.append(np.random.choice(attack_cats[1:]))
        
        df['binary_label'] = labels
        df['attack_type'] = attack_categories
    
    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'nf_{dataset_name}.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Sample NetFlow {dataset_name} dataset saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate example data for testing')
    parser.add_argument('--output_dir', default='temp', help='Output directory for sample datasets')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples per dataset')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate standard dataset samples
    generate_bot_iot_sample(args.output_dir, args.num_samples)
    generate_cic_ids2018_sample(args.output_dir, args.num_samples)
    generate_unsw_nb15_sample(args.output_dir, args.num_samples)
    
    # Generate netflow dataset samples
    generate_netflow_sample(args.output_dir, 'bot_iot', args.num_samples)
    generate_netflow_sample(args.output_dir, 'cic_ids2018', args.num_samples)
    generate_netflow_sample(args.output_dir, 'unsw_nb15', args.num_samples)
    
    print(f"\nAll sample datasets have been generated in {args.output_dir}")
    print("You can now run standardize_datasets.py on these samples:")
    print(f"python standardize_datasets.py --input_dir {args.output_dir} --standard_output_dir standard --netflow_output_dir netflow")

if __name__ == '__main__':
    main() 