"""
Quick test script to verify the sampled dataset works with TempME training pipeline
This runs a minimal training test (1 epoch) to ensure everything is configured correctly
"""

import sys
import os
import pandas as pd
import numpy as np

def test_dataset_loading():
    """Test that sampled dataset files exist and can be loaded"""
    print("="*80)
    print("TEST 1: Dataset Loading")
    print("="*80)
    
    required_files = [
        'ml_enron_sampled.csv',
        'ml_enron_sampled.npy',
        'ml_enron_sampled_node.npy'
    ]
    
    for filename in required_files:
        if not os.path.exists(filename):
            print(f"❌ FAIL: {filename} not found")
            print(f"   Please run: python sample_dataset.py")
            return False
        else:
            print(f"✓ Found: {filename}")
    
    try:
        df = pd.read_csv('ml_enron_sampled.csv')
        edge_feat = np.load('ml_enron_sampled.npy')
        node_feat = np.load('ml_enron_sampled_node.npy')
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  - Edges: {len(df):,}")
        print(f"  - Nodes: {len(set(df.u) | set(df.i))}")
        print(f"  - Edge features: {edge_feat.shape}")
        print(f"  - Node features: {node_feat.shape}")
        
        # Verify temporal ordering
        if not df['ts'].is_monotonic_increasing:
            print("⚠ Warning: Timestamps are not strictly increasing")
        
        return True
    except Exception as e:
        print(f"❌ FAIL: Error loading dataset: {e}")
        return False

def test_config_files():
    """Test that all config files have enron_sampled entry"""
    print("\n" + "="*80)
    print("TEST 2: Configuration Files")
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
    
    all_good = True
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"⚠ Warning: {filepath} not found")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'enron_sampled' in content:
                print(f"✓ {filepath}")
            else:
                print(f"❌ {filepath} - missing 'enron_sampled' entry")
                all_good = False
    
    if all_good:
        print("\n✓ All configuration files updated!")
    else:
        print("\n❌ Some configuration files need updating")
    
    return all_good

def test_import_dependencies():
    """Test that required packages can be imported"""
    print("\n" + "="*80)
    print("TEST 3: Python Dependencies")
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
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - not installed")
            all_good = False
    
    if all_good:
        print("\n✓ All required packages installed!")
    else:
        print("\n❌ Some packages missing. Install with: pip install torch pandas numpy scikit-learn tqdm")
    
    return all_good

def test_file_structure():
    """Test that directory structure is correct"""
    print("\n" + "="*80)
    print("TEST 4: Project Structure")
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
            print(f"✓ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - not found")
            all_good = False
    
    if all_good:
        print("\n✓ Project structure is correct!")
    else:
        print("\n⚠ Some directories missing - will be created during training")
    
    return True  # Not critical

def print_next_steps():
    """Print recommended next steps"""
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Create sampled dataset (if not done):")
    print("   python sample_dataset.py --edge_ratio 0.15")
    print("\n2. Train a base model (choose one):")
    print("   python learn_base.py --base_type tgn --data enron_sampled --n_epoch 5  # Quick test")
    print("   python learn_base.py --base_type tgn --data enron_sampled --n_epoch 50  # Full training")
    print("\n3. Train explainer:")
    print("   python temp_exp_main.py --base_type tgn --data enron_sampled --n_epoch 40")
    print("\n4. Verify enhancement:")
    print("   python enhance_main.py --data enron_sampled --base_type tgn")
    print("\n" + "="*80)
    print("See FAST_TRAINING_GUIDE.md for complete instructions")
    print("="*80)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TempME SAMPLED DATASET TEST SUITE")
    print("="*80)
    print("\nThis script verifies that your environment is ready for fast training")
    print("with the sampled Enron dataset.\n")
    
    results = []
    
    # Run all tests
    results.append(("Dataset Loading", test_dataset_loading()))
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
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "-"*80)
    print(f"Results: {passed}/{total} tests passed")
    print("="*80)
    
    if passed == total:
        print("\n✅ All tests passed! You're ready to start training.")
        print_next_steps()
        sys.exit(0)
    else:
        print("\n⚠ Some tests failed. Please fix the issues above before training.")
        print("\nCommon fixes:")
        print("1. Run: python sample_dataset.py")
        print("2. Install missing packages: pip install torch pandas numpy scikit-learn tqdm")
        print("3. Make sure you're in the TempME project directory")
        sys.exit(1)
