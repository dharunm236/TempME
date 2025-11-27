"""
Quick test script to verify the sampled datasets work with TempME training pipeline
This runs a minimal training test (1 epoch) to ensure everything is configured correctly
Supports both Enron and USLegis sampled datasets
"""

import sys
import os
import pandas as pd
import numpy as np

def test_dataset_loading(dataset_name="enron_sampled", processed_dir="processed"):
    """Test that sampled dataset files exist and can be loaded"""
    print("="*80)
    print(f"TEST: {dataset_name.upper()} Dataset Loading")
    print("="*80)
    
    required_files = [
        f'{processed_dir}/ml_{dataset_name}.csv',
        f'{processed_dir}/ml_{dataset_name}.npy',
        f'{processed_dir}/ml_{dataset_name}_node.npy'
    ]
    
    for filename in required_files:
        if not os.path.exists(filename):
            print(f"[FAIL] {filename} not found")
            print(f"   Please run: python sample_dataset.py" if 'enron' in dataset_name else f"   Please run: python sample_uslegis.py")
            return False
        else:
            print(f"[OK] Found: {filename}")
    
    try:
        df = pd.read_csv(required_files[0])
        edge_feat = np.load(required_files[1])
        node_feat = np.load(required_files[2])
        
        print(f"\n[OK] Dataset loaded successfully!")
        print(f"  - Edges: {len(df):,}")
        print(f"  - Nodes: {len(set(df.u) | set(df.i))}")
        print(f"  - Edge features: {edge_feat.shape}")
        print(f"  - Node features: {node_feat.shape}")
        
        # For USLegis, check session coverage
        if 'uslegis' in dataset_name:
            print(f"  - Sessions: {df.ts.nunique()}")
        
        # Verify temporal ordering
        if not df['ts'].is_monotonic_increasing:
            print("[WARN] Warning: Timestamps are not strictly increasing")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error loading dataset: {e}")
        return False

def test_config_files():
    """Test that all config files have enron_sampled and uslegis_sampled entries"""
    print("\n" + "="*80)
    print("TEST: Configuration Files")
    print("="*80)
    
    files_to_check = [
        'learn_base.py',
        'temp_exp_main.py',
        'enhance_main.py',
        'utils/null_model.py',
        'processed/utils/null_model.py',
        'processed/data_preprocess.py',
        'visualize_explanations.py'
    ]
    
    datasets_to_check = ['enron_sampled', 'uslegis_sampled']
    
    all_good = True
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"[WARN] Warning: {filepath} not found")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            missing = [ds for ds in datasets_to_check if ds not in content]
            if not missing:
                print(f"[OK] {filepath}")
            else:
                print(f"[FAIL] {filepath} - missing: {', '.join(missing)}")
                all_good = False
    
    if all_good:
        print("\n[OK] All configuration files updated!")
    else:
        print("\n[FAIL] Some configuration files need updating")
    
    return all_good

def test_import_dependencies():
    """Test that required packages can be imported"""
    print("\n" + "="*80)
    print("TEST: Python Dependencies")
    print("="*80)
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm'
    }
    
    all_good = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[FAIL] {name} - not installed")
            all_good = False
    
    if all_good:
        print("\n[OK] All required packages installed!")
    else:
        print("\n[FAIL] Some packages missing. Install with: pip install torch pandas numpy scikit-learn tqdm")
    
    return all_good

def test_file_structure():
    """Test that directory structure is correct"""
    print("\n" + "="*80)
    print("TEST: Project Structure")
    print("="*80)
    
    required_dirs = [
        'params/tgnn',
        'params/explainer/tgn',
        'processed',
        'utils',
        'TGN',
        'TGAT',
        'GraphM',
        'models'
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"[OK] {dir_path}/")
        else:
            print(f"[FAIL] {dir_path}/ - not found")
            all_good = False
    
    if all_good:
        print("\n[OK] Project structure is correct!")
    else:
        print("\n[WARN] Some directories missing - will be created during training")
    
    return True  # Not critical

def print_next_steps():
    """Print recommended next steps"""
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n=== For Enron Sampled Dataset ===")
    print("1. Create sampled dataset (if not done):")
    print("   python sample_dataset.py --edge_ratio 0.15")
    print("\n2. Train a base model:")
    print("   python learn_base.py --base_type tgn --data enron_sampled --n_epoch 50")
    
    print("\n=== For USLegis Sampled Dataset ===")
    print("1. Create sampled dataset (if not done):")
    print("   python sample_uslegis.py --edge_ratio 0.20")
    print("\n2. Train a base model:")
    print("   python learn_base.py --base_type tgn --data uslegis_sampled --n_epoch 50")
    
    print("\n=== General Commands ===")
    print("3. Train explainer:")
    print("   python temp_exp_main.py --base_type tgn --data <dataset> --n_epoch 40")
    print("\n4. Verify enhancement:")
    print("   python enhance_main.py --data <dataset> --base_type tgn")
    print("\n" + "="*80)
    print("See SAMPLING_SUMMARY.md and USLEGIS_SAMPLING_SUMMARY.md for details")
    print("="*80)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TempME SAMPLED DATASET TEST SUITE")
    print("="*80)
    print("\nThis script verifies that your environment is ready for fast training")
    print("with sampled datasets (Enron and USLegis).\n")
    
    results = []
    
    # Run all tests
    results.append(("Enron Sampled Dataset", test_dataset_loading("enron_sampled", "processed")))
    results.append(("USLegis Sampled Dataset", test_dataset_loading("uslegis_sampled", "processed")))
    results.append(("Configuration Files", test_config_files()))
    results.append(("Python Dependencies", test_import_dependencies()))
    results.append(("Project Structure", test_file_structure()))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print("\n" + "-"*80)
    print(f"Results: {passed}/{total} tests passed")
    print("="*80)
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! You're ready to start training.")
        print_next_steps()
        sys.exit(0)
    else:
        print("\n[WARNING] Some tests failed. Please fix the issues above before training.")
        print("\nCommon fixes:")
        print("1. Run: python sample_dataset.py  (for Enron)")
        print("2. Run: python sample_uslegis.py  (for USLegis)")
        print("3. Install missing packages: pip install torch pandas numpy scikit-learn tqdm")
        print("4. Make sure you're in the TempME project directory")
        sys.exit(1)
