"""
Compute Node Degrees from Graph for Explainer_new.py
====================================================

This script computes actual node degrees from the preprocessed graph data
and updates the explainer's node_degree tensor.

Usage:
    python compute_node_degrees.py --data enron_sampled --explainer_path params/explainer/tgn/enron_sampled.pt
"""

import argparse
import numpy as np
import pandas as pd
import torch
import os.path as osp
from collections import defaultdict


def compute_node_degrees_from_edgelist(data_name):
    """
    Compute node degrees from the CSV edge list
    
    Returns:
        torch.Tensor: Node degrees indexed by node ID
    """
    print(f"\n{'='*80}")
    print(f"Computing Node Degrees for {data_name}")
    print(f"{'='*80}\n")
    
    # Load edge list
    csv_path = osp.join('processed', f'ml_{data_name}.csv')
    if not osp.exists(csv_path):
        csv_path = f'ml_{data_name}.csv'
    
    print(f"Loading edge list from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    src_l = df.u.values
    dst_l = df.i.values
    
    print(f"Total edges: {len(df):,}")
    print(f"Unique source nodes: {len(set(src_l)):,}")
    print(f"Unique destination nodes: {len(set(dst_l)):,}")
    
    # Find max node ID
    max_node_id = max(src_l.max(), dst_l.max())
    num_nodes = max_node_id + 1
    
    print(f"Max node ID: {max_node_id}")
    print(f"Total nodes (including isolated): {num_nodes}")
    
    # Count degrees (undirected)
    degree_dict = defaultdict(int)
    for src, dst in zip(src_l, dst_l):
        degree_dict[src] += 1
        degree_dict[dst] += 1
    
    # Create degree tensor
    degrees = torch.ones(num_nodes, dtype=torch.float32)  # Default 1 for isolated nodes
    for node_id, degree in degree_dict.items():
        degrees[node_id] = float(degree)
    
    # Statistics
    print(f"\nDegree Statistics:")
    print(f"  Min degree: {degrees.min().item():.0f}")
    print(f"  Max degree: {degrees.max().item():.0f}")
    print(f"  Mean degree: {degrees.mean().item():.2f}")
    print(f"  Median degree: {degrees.median().item():.0f}")
    print(f"  Std degree: {degrees.std().item():.2f}")
    
    # Top 10 high-degree nodes
    top_k = 10
    top_degrees, top_nodes = torch.topk(degrees, min(top_k, num_nodes))
    print(f"\nTop {top_k} high-degree nodes:")
    for i, (node, deg) in enumerate(zip(top_nodes.tolist(), top_degrees.tolist())):
        print(f"  {i+1}. Node {node}: degree {deg:.0f}")
    
    return degrees


def update_explainer_node_degrees(explainer, node_degrees):
    """
    Update the node_degree tensor in an explainer model
    """
    print(f"\n{'='*80}")
    print("Updating Explainer Node Degrees")
    print(f"{'='*80}\n")
    
    # Check current state
    current_degrees = explainer.node_degree
    print(f"Current node_degree tensor shape: {current_degrees.shape}")
    print(f"New node_degree tensor shape: {node_degrees.shape}")
    
    if current_degrees.shape[0] != node_degrees.shape[0]:
        print(f"⚠️  Warning: Shape mismatch!")
        print(f"   Explainer expects {current_degrees.shape[0]} nodes")
        print(f"   Graph has {node_degrees.shape[0]} nodes")
        
        # Pad or truncate as needed
        if node_degrees.shape[0] < current_degrees.shape[0]:
            print(f"   Padding with ones to match explainer size")
            padded = torch.ones(current_degrees.shape[0])
            padded[:node_degrees.shape[0]] = node_degrees
            node_degrees = padded
        else:
            print(f"   Truncating to match explainer size")
            node_degrees = node_degrees[:current_degrees.shape[0]]
    
    # Update
    explainer.node_degree = node_degrees.to(explainer.node_degree.device)
    
    # Verify
    print(f"✅ Updated node_degree tensor")
    print(f"   Device: {explainer.node_degree.device}")
    print(f"   Min: {explainer.node_degree.min().item():.0f}")
    print(f"   Max: {explainer.node_degree.max().item():.0f}")
    print(f"   Mean: {explainer.node_degree.mean().item():.2f}")
    
    return explainer


def save_node_degrees(node_degrees, data_name):
    """
    Save computed node degrees to file for reuse
    """
    output_path = f'processed/{data_name}_node_degrees.pt'
    torch.save(node_degrees, output_path)
    print(f"\n✅ Saved node degrees to: {output_path}")
    print(f"   Load with: node_degrees = torch.load('{output_path}')")
    return output_path


def load_node_degrees(data_name):
    """
    Load previously computed node degrees
    """
    degree_path = f'processed/{data_name}_node_degrees.pt'
    if osp.exists(degree_path):
        print(f"Loading cached node degrees from: {degree_path}")
        return torch.load(degree_path)
    return None


def main():
    parser = argparse.ArgumentParser('Compute Node Degrees for Explainer')
    parser.add_argument('--data', type=str, default='enron_sampled',
                       help='Dataset name (e.g., enron, enron_sampled)')
    parser.add_argument('--explainer_path', type=str, default=None,
                       help='Path to trained explainer model (optional)')
    parser.add_argument('--force_recompute', action='store_true',
                       help='Force recomputation even if cached file exists')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("NODE DEGREE COMPUTATION TOOL")
    print("="*80)
    
    # Try to load cached degrees first
    if not args.force_recompute:
        cached_degrees = load_node_degrees(args.data)
        if cached_degrees is not None:
            print("✅ Using cached node degrees")
            node_degrees = cached_degrees
        else:
            node_degrees = compute_node_degrees_from_edgelist(args.data)
            save_node_degrees(node_degrees, args.data)
    else:
        node_degrees = compute_node_degrees_from_edgelist(args.data)
        save_node_degrees(node_degrees, args.data)
    
    # Update explainer if path provided
    if args.explainer_path:
        print(f"\n⚠️  Note: Loading and updating explainer model")
        print(f"   This requires the full model architecture to be available")
        print(f"   You may need to modify this script to match your exact setup")
        
        # This part requires your specific model architecture
        # Uncomment and modify as needed:
        """
        from TGN.tgn import TGN
        from models.explainer_new import TempME
        
        # Load base model
        base_model = TGN(...)  # Add your parameters
        base_model.load_state_dict(torch.load('params/tgnn/tgn_enron_sampled.pt'))
        
        # Load explainer
        explainer = TempME(base_model, ...)  # Add your parameters
        explainer.load_state_dict(torch.load(args.explainer_path))
        
        # Update degrees
        explainer = update_explainer_node_degrees(explainer, node_degrees)
        
        # Save updated model
        torch.save(explainer.state_dict(), args.explainer_path)
        print(f"✅ Saved updated explainer to: {args.explainer_path}")
        """
    
    print("\n" + "="*80)
    print("USAGE IN TRAINING CODE:")
    print("="*80)
    print("""
# In your training script (temp_exp_main.py or enhance_main.py):

import torch

# After creating explainer
explainer = TempME(base_model, ...)

# Load computed node degrees
node_degrees = torch.load(f'processed/{args.data}_node_degrees.pt')
explainer.node_degree = node_degrees.to(args.device)

print(f"Loaded node degrees: min={explainer.node_degree.min():.0f}, max={explainer.node_degree.max():.0f}")

# Continue training...
""")
    
    print("\n✅ Node degree computation complete!")


if __name__ == "__main__":
    main()
