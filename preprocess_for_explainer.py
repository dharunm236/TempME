"""
Preprocess dataset for TempME explainer training
This script generates the required H5 files for a specific dataset
"""

import argparse
import os
import sys

def preprocess_dataset(dataset_name):
    """
    Preprocess a dataset by updating data_preprocess.py and running it
    """
    print(f"\n{'='*80}")
    print(f"Preprocessing dataset: {dataset_name}")
    print(f"{'='*80}\n")
    
    preprocess_file = os.path.join('processed', 'data_preprocess.py')
    
    if not os.path.exists(preprocess_file):
        print(f"❌ Error: {preprocess_file} not found")
        return False
    
    # Read the file
    with open(preprocess_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update the dataset name in the for loop
    # Find the line: for data in ["enron"]:
    import re
    pattern = r'for data in \["[^"]+"\]:'
    replacement = f'for data in ["{dataset_name}"]:'
    
    if not re.search(pattern, content):
        print(f"⚠ Warning: Could not find dataset loop pattern in {preprocess_file}")
        print("Please manually edit the file to set the dataset name")
        return False
    
    content = re.sub(pattern, replacement, content)
    
    # Update the global data variable at the top
    pattern2 = r'data = "[^"]+"'
    replacement2 = f'data = "{dataset_name}"'
    content = re.sub(pattern2, replacement2, content)
    
    # Write back
    with open(preprocess_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Updated {preprocess_file} to process '{dataset_name}'")
    print(f"\nStarting preprocessing...")
    print(f"This will generate:")
    print(f"  - processed/{dataset_name}_train.h5")
    print(f"  - processed/{dataset_name}_test.h5")
    print(f"  - processed/{dataset_name}_train_cat.h5")
    print(f"  - processed/{dataset_name}_test_cat.h5")
    print(f"  - processed/{dataset_name}_train_edge.npy")
    print(f"  - processed/{dataset_name}_test_edge.npy")
    print(f"\nThis may take 5-10 minutes depending on dataset size...")
    print(f"{'='*80}\n")
    
    # Run the preprocessing
    os.chdir('processed')
    exit_code = os.system('python data_preprocess.py')
    os.chdir('..')
    
    if exit_code == 0:
        print(f"\n{'='*80}")
        print(f"✅ Preprocessing completed successfully!")
        print(f"{'='*80}")
        return True
    else:
        print(f"\n{'='*80}")
        print(f"❌ Preprocessing failed with exit code {exit_code}")
        print(f"{'='*80}")
        return False


def check_preprocessing_files(dataset_name):
    """
    Check if preprocessing files exist for a dataset
    """
    required_files = [
        f'processed/{dataset_name}_train_cat.h5',
        f'processed/{dataset_name}_test_cat.h5',
        f'processed/{dataset_name}_train_edge.npy',
        f'processed/{dataset_name}_test_edge.npy'
    ]
    
    missing = []
    existing = []
    
    for filepath in required_files:
        if os.path.exists(filepath):
            existing.append(filepath)
        else:
            missing.append(filepath)
    
    print(f"\n{'='*80}")
    print(f"Preprocessing Status for '{dataset_name}'")
    print(f"{'='*80}\n")
    
    if existing:
        print("✓ Existing files:")
        for f in existing:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")
    
    if missing:
        print("\n✗ Missing files:")
        for f in missing:
            print(f"  - {f}")
        print(f"\nRun: python preprocess_for_explainer.py --dataset {dataset_name}")
        return False
    else:
        print(f"\n✅ All preprocessing files exist for '{dataset_name}'")
        print("You can now train the explainer!")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess dataset for TempME explainer training')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., enron, enron_sampled)')
    parser.add_argument('--check', action='store_true',
                       help='Only check if preprocessing files exist')
    
    args = parser.parse_args()
    
    if args.check:
        check_preprocessing_files(args.dataset)
    else:
        # Check if dataset files exist
        csv_file = f'processed/ml_{args.dataset}.csv'
        npy_file = f'processed/ml_{args.dataset}.npy'
        node_npy_file = f'processed/ml_{args.dataset}_node.npy'
        
        if not os.path.exists(csv_file):
            print(f"❌ Error: Dataset file not found: {csv_file}")
            print(f"Please ensure the dataset '{args.dataset}' exists in the processed/ folder")
            sys.exit(1)
        
        if not os.path.exists(npy_file):
            print(f"❌ Error: Edge features file not found: {npy_file}")
            sys.exit(1)
        
        if not os.path.exists(node_npy_file):
            print(f"❌ Error: Node features file not found: {node_npy_file}")
            sys.exit(1)
        
        print(f"✓ Dataset files found for '{args.dataset}'")
        
        # Preprocess
        success = preprocess_dataset(args.dataset)
        
        if success:
            print(f"\n{'='*80}")
            print("NEXT STEPS")
            print(f"{'='*80}")
            print(f"\n1. Train explainer:")
            print(f"   python temp_exp_main.py --base_type tgn --data {args.dataset}")
            print(f"\n2. Verify enhancement:")
            print(f"   python enhance_main.py --data {args.dataset} --base_type tgn")
            print(f"\n{'='*80}")
            sys.exit(0)
        else:
            sys.exit(1)
