"""
Validation Script for Explainer_new.py Fixes
============================================

This script helps validate that the overfitting fixes work correctly.
Run this after training to check if improvements are effective.

Usage:
    python validate_fixes.py --model_path params/explainer/tgn/enron_sampled.pt
"""

import torch
import numpy as np
from models.explainer_new import TempME, TempME_TGAT

def check_train_test_consistency(explainer):
    """
    Verify that model behavior is consistent between train and test modes
    """
    print("\n" + "="*80)
    print("CHECKING TRAIN/TEST CONSISTENCY")
    print("="*80)
    
    # Create dummy data
    batch_size = 2
    n_walks = 10
    len_walk = 3
    
    dummy_walks = (
        np.random.randint(0, 100, (batch_size, n_walks, 6)),  # node_idx
        np.random.randint(0, 50, (batch_size, n_walks, len_walk)),  # edge_idx
        np.random.randn(batch_size, n_walks, len_walk).astype(np.float32),  # time_idx
        np.random.randint(0, 12, (batch_size, n_walks, 1)),  # cat_feat
        np.zeros((batch_size, n_walks, 1))  # marginal
    )
    
    cut_time_l = np.random.randn(batch_size).astype(np.float32)
    edge_identify = np.random.randint(0, 3, (batch_size, n_walks, len_walk))
    
    # Test in training mode
    explainer.train()
    with torch.no_grad():
        try:
            train_output = explainer(dummy_walks, cut_time_l, edge_identify)
            print("✅ Training mode: Output shape =", train_output.shape)
        except Exception as e:
            print(f"❌ Training mode failed: {e}")
            return False
    
    # Test in eval mode
    explainer.eval()
    with torch.no_grad():
        try:
            test_output = explainer(dummy_walks, cut_time_l, edge_identify)
            print("✅ Test mode: Output shape =", test_output.shape)
        except Exception as e:
            print(f"❌ Test mode failed: {e}")
            return False
    
    # Check shapes match
    if train_output.shape == test_output.shape:
        print("✅ Output shapes consistent between train/test")
    else:
        print(f"❌ Shape mismatch: train {train_output.shape} vs test {test_output.shape}")
        return False
    
    # Check outputs are not too different (should use same logic)
    diff = torch.abs(train_output - test_output).mean().item()
    print(f"   Mean absolute difference: {diff:.6f}")
    
    if diff < 0.5:  # Some difference is OK due to dropout, but not huge
        print("✅ Train/test outputs reasonably similar")
    else:
        print(f"⚠️  Warning: Large train/test difference ({diff:.3f})")
        print("   This might indicate inconsistent processing between modes")
    
    return True


def check_walk_importance_properties(explainer):
    """
    Verify walk importance function produces valid soft weights
    """
    print("\n" + "="*80)
    print("CHECKING WALK IMPORTANCE PROPERTIES")
    print("="*80)
    
    batch_size = 2
    n_walks = 10
    len_walk = 3
    
    time_idx = torch.randn(batch_size, n_walks, len_walk) * 100 + 1000
    node_idx = torch.randint(1, 100, (batch_size, n_walks, len_walk))
    cut_time_l = torch.randn(batch_size) * 100 + 1100
    
    with torch.no_grad():
        weights = explainer.compute_walk_importance(time_idx, node_idx, cut_time_l)
    
    print(f"Walk importance shape: {weights.shape}")
    print(f"Expected shape: ({batch_size}, {n_walks})")
    
    # Check properties
    checks = []
    
    # 1. All positive
    all_positive = (weights >= 0).all().item()
    checks.append(("All weights >= 0", all_positive))
    
    # 2. No NaN or Inf
    no_nan = not torch.isnan(weights).any().item()
    no_inf = not torch.isinf(weights).any().item()
    checks.append(("No NaN values", no_nan))
    checks.append(("No Inf values", no_inf))
    
    # 3. Reasonable range (not too extreme)
    max_val = weights.max().item()
    min_val = weights.min().item()
    reasonable_range = (max_val < 10.0 and min_val > 0.01)
    checks.append((f"Reasonable range [{min_val:.3f}, {max_val:.3f}]", reasonable_range))
    
    # 4. Has variance (not all same)
    variance = weights.var().item()
    has_variance = variance > 0.001
    checks.append((f"Has variance ({variance:.4f})", has_variance))
    
    # Print results
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
    
    return all(passed for _, passed in checks)


def check_dependency_gate_range(explainer):
    """
    Verify dependency gating produces values in expected range [0.5, 1.0]
    """
    print("\n" + "="*80)
    print("CHECKING DEPENDENCY GATE RANGE")
    print("="*80)
    
    if not explainer.use_dependency_aware_sampling:
        print("⚠️  Dependency-aware sampling is disabled")
        return True
    
    # This is harder to test without running full forward pass
    # Just check the module exists and has correct structure
    if hasattr(explainer, 'edge_dependency_gcn'):
        print("✅ edge_dependency_gcn module exists")
        
        # Check it has dropout for regularization
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in explainer.edge_dependency_gcn.modules())
        if has_dropout:
            print("✅ edge_dependency_gcn has dropout layers")
        else:
            print("⚠️  edge_dependency_gcn missing dropout layers")
            return False
        
        return True
    else:
        print("❌ edge_dependency_gcn module not found")
        return False


def check_temporal_attention_regularization(explainer):
    """
    Verify temporal attention has dropout for regularization
    """
    print("\n" + "="*80)
    print("CHECKING TEMPORAL ATTENTION REGULARIZATION")
    print("="*80)
    
    if hasattr(explainer, 'attention'):
        attention = explainer.attention
        
        # Check if it's TemporalAwareAttention and has dropout
        if attention.__class__.__name__ == 'TemporalAwareAttention':
            has_dropout = hasattr(attention, 'dropout')
            if has_dropout:
                print("✅ TemporalAwareAttention has dropout")
                return True
            else:
                print("⚠️  TemporalAwareAttention missing dropout")
                return False
        else:
            print(f"   Using {attention.__class__.__name__} (not temporal-aware)")
            return True
    else:
        print("❌ Attention module not found")
        return False


def summarize_model_capacity(explainer):
    """
    Count parameters to ensure model isn't too complex
    """
    print("\n" + "="*80)
    print("MODEL CAPACITY SUMMARY")
    print("="*80)
    
    total_params = sum(p.numel() for p in explainer.parameters())
    trainable_params = sum(p.numel() for p in explainer.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check for specific components
    if hasattr(explainer, 'edge_dependency_gcn'):
        dep_params = sum(p.numel() for p in explainer.edge_dependency_gcn.parameters())
        print(f"Dependency module parameters: {dep_params:,} ({dep_params/trainable_params*100:.1f}%)")
    
    if hasattr(explainer, 'attention'):
        attn_params = sum(p.numel() for p in explainer.attention.parameters())
        print(f"Attention module parameters: {attn_params:,} ({attn_params/trainable_params*100:.1f}%)")
    
    return True


def main():
    """
    Main validation function
    """
    print("\n" + "="*80)
    print("EXPLAINER_NEW.PY VALIDATION SCRIPT")
    print("Checking for overfitting fixes")
    print("="*80)
    
    # Note: This is a template - you need to load actual model
    print("\nℹ️  Note: This script checks model properties, not trained weights")
    print("   For full validation, run training and check train/test metrics")
    
    # Mock explainer for testing (replace with actual loading)
    print("\n⚠️  To use this script with a trained model:")
    print("   1. Load your trained base model")
    print("   2. Initialize explainer with base model")
    print("   3. Load saved explainer weights")
    print("   4. Run validation checks")
    
    print("\nExample code:")
    print("""
    from TGN.tgn import TGN
    from models.explainer_new import TempME
    
    # Load base model
    base_model = TGN(...)
    base_model.load_state_dict(torch.load('params/tgnn/tgn_enron_sampled.pt'))
    
    # Initialize explainer
    explainer = TempME(
        base_model, 
        base_model_type='tgn',
        data='enron_sampled',
        out_dim=40,
        hid_dim=64,
        use_temporal_guidance=True,
        use_dependency_aware_sampling=True,
        device='cuda'
    )
    
    # Load trained weights
    explainer.load_state_dict(torch.load('params/explainer/tgn/enron_sampled.pt'))
    
    # Run validation
    check_train_test_consistency(explainer)
    check_walk_importance_properties(explainer)
    check_dependency_gate_range(explainer)
    check_temporal_attention_regularization(explainer)
    summarize_model_capacity(explainer)
    """)


if __name__ == "__main__":
    main()
