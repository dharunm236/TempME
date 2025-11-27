"""

SPECIFICALLY MADE FOR ENRON DATASET

Dataset Sampling Script for TempME
This script creates a sampled version of the Enron dataset that maintains:
1. Temporal ordering and patterns
2. Graph structural properties
3. Node importance distribution
4. Sufficient data for training TGNN models quickly

The sampled dataset should enable training within 12 hours while preserving
research validity for publication purposes.
"""

import pandas as pd
import numpy as np
import os
import argparse
from collections import defaultdict, Counter


def sample_dataset(input_csv, input_npy, input_node_npy, output_prefix, 
                   edge_sample_ratio=0.15, temporal_windows=10, min_node_degree=2,
                   seed=42):
    """
    Sample the dataset intelligently to maintain temporal and structural properties.
    
    Args:
        input_csv: Path to original CSV file
        input_npy: Path to original edge features
        input_node_npy: Path to original node features
        output_prefix: Prefix for output files (e.g., 'ml_enron_sampled')
        edge_sample_ratio: Ratio of edges to keep (0.15 = 15%)
        temporal_windows: Number of temporal windows for stratified sampling
        min_node_degree: Minimum degree to keep a node
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    print("="*80)
    print("TempME Dataset Sampling Tool")
    print("="*80)
    
    # Load data
    print("\n[1/7] Loading original dataset...")
    df = pd.read_csv(input_csv)
    edge_feat = np.load(input_npy)
    node_feat = np.load(input_node_npy)
    
    print(f"  Original dataset statistics:")
    print(f"  - Total edges: {len(df):,}")
    print(f"  - Unique nodes: {len(set(df.u.values) | set(df.i.values))}")
    print(f"  - Edge features shape: {edge_feat.shape}")
    print(f"  - Node features shape: {node_feat.shape}")
    print(f"  - Time range: {df.ts.min():.0f} to {df.ts.max():.0f}")
    
    # Step 1: Temporal stratified sampling
    print(f"\n[2/7] Performing temporal stratified sampling...")
    df['time_window'] = pd.qcut(df.ts, q=temporal_windows, labels=False, duplicates='drop')
    
    # Calculate edges per window
    edges_per_window = len(df) * edge_sample_ratio / temporal_windows
    
    sampled_indices = []
    for window in range(temporal_windows):
        window_df = df[df['time_window'] == window]
        n_samples = min(int(edges_per_window), len(window_df))
        window_indices = np.random.choice(window_df.index, size=n_samples, replace=False)
        sampled_indices.extend(window_indices)
    
    df_sampled = df.loc[sampled_indices].copy()
    print(f"  - Sampled {len(df_sampled):,} edges from {temporal_windows} temporal windows")
    
    # Step 2: Identify active nodes and their importance
    print(f"\n[3/7] Analyzing node importance...")
    node_degree = defaultdict(int)
    for idx, row in df_sampled.iterrows():
        node_degree[row['u']] += 1
        node_degree[row['i']] += 1
    
    # Keep nodes with sufficient activity
    active_nodes = {node for node, deg in node_degree.items() if deg >= min_node_degree}
    print(f"  - Active nodes (degree >= {min_node_degree}): {len(active_nodes)}")
    
    # Step 3: Filter edges to only include active nodes
    print(f"\n[4/7] Filtering edges for active nodes...")
    df_sampled = df_sampled[
        df_sampled['u'].isin(active_nodes) & df_sampled['i'].isin(active_nodes)
    ].copy()
    print(f"  - Remaining edges: {len(df_sampled):,}")
    
    # Step 4: Remap node IDs to be contiguous
    print(f"\n[5/7] Remapping node IDs...")
    unique_nodes = sorted(set(df_sampled.u.values) | set(df_sampled.i.values))
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    
    df_sampled['u'] = df_sampled['u'].map(node_mapping)
    df_sampled['i'] = df_sampled['i'].map(node_mapping)
    
    print(f"  - Remapped {len(unique_nodes)} nodes to IDs 0-{len(unique_nodes)-1}")
    
    # Step 5: Reindex edges
    df_sampled = df_sampled.sort_values('ts').reset_index(drop=True)
    df_sampled['idx'] = np.arange(1, len(df_sampled) + 1)
    df_sampled['Unnamed: 0'] = np.arange(len(df_sampled))
    
    # Step 6: Sample corresponding features
    print(f"\n[6/7] Sampling edge and node features...")
    original_indices = sampled_indices
    edge_feat_sampled = edge_feat[original_indices]
    
    # Sample node features for active nodes
    node_feat_sampled = node_feat[unique_nodes]
    
    print(f"  - Edge features shape: {edge_feat_sampled.shape}")
    print(f"  - Node features shape: {node_feat_sampled.shape}")
    
    # Step 7: Save sampled dataset
    print(f"\n[7/7] Saving sampled dataset...")
    df_sampled.to_csv(f'{output_prefix}.csv', index=False)
    np.save(f'{output_prefix}.npy', edge_feat_sampled)
    np.save(f'{output_prefix}_node.npy', node_feat_sampled)
    
    print(f"  - Saved: {output_prefix}.csv")
    print(f"  - Saved: {output_prefix}.npy")
    print(f"  - Saved: {output_prefix}_node.npy")
    
    # Print final statistics
    print("\n" + "="*80)
    print("SAMPLING SUMMARY")
    print("="*80)
    print(f"Original dataset:")
    print(f"  - Edges: {len(df):,}")
    print(f"  - Nodes: {len(set(df.u.values) | set(df.i.values))}")
    print(f"  - Edge features size: {edge_feat.nbytes / 1024 / 1024:.2f} MB")
    print(f"\nSampled dataset:")
    print(f"  - Edges: {len(df_sampled):,} ({len(df_sampled)/len(df)*100:.2f}%)")
    print(f"  - Nodes: {len(unique_nodes)} ({len(unique_nodes)/len(set(df.u.values) | set(df.i.values))*100:.2f}%)")
    print(f"  - Edge features size: {edge_feat_sampled.nbytes / 1024 / 1024:.2f} MB")
    print(f"  - Compression ratio: {len(df_sampled)/len(df):.3f}x")
    print(f"\nEstimated training speedup: {len(df)/len(df_sampled):.1f}x faster")
    print("="*80)
    
    # Validate temporal ordering
    assert df_sampled['ts'].is_monotonic_increasing or (df_sampled['ts'].diff()[1:] >= 0).all(), \
        "Temporal ordering violated!"
    
    print("\n✓ Temporal ordering preserved")
    print("✓ Sampled dataset created successfully!")
    print("\nYou can now train models using this sampled dataset.")
    print(f"Example: python learn_base.py --base_type tgn --data {os.path.basename(output_prefix).replace('ml_', '')}")
    
    return df_sampled, edge_feat_sampled, node_feat_sampled


def analyze_dataset(csv_path):
    """Analyze and print statistics about a dataset."""
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*80)
    print(f"Dataset Analysis: {csv_path}")
    print("="*80)
    
    print(f"\nBasic Statistics:")
    print(f"  - Total edges: {len(df):,}")
    print(f"  - Unique source nodes: {df['u'].nunique()}")
    print(f"  - Unique destination nodes: {df['i'].nunique()}")
    print(f"  - Total unique nodes: {len(set(df.u.values) | set(df.i.values))}")
    print(f"  - Time span: {df.ts.max() - df.ts.min():.0f}")
    
    # Degree distribution
    node_degree = defaultdict(int)
    for idx, row in df.iterrows():
        node_degree[row['u']] += 1
        node_degree[row['i']] += 1
    
    degrees = list(node_degree.values())
    print(f"\nDegree Statistics:")
    print(f"  - Mean degree: {np.mean(degrees):.2f}")
    print(f"  - Median degree: {np.median(degrees):.2f}")
    print(f"  - Max degree: {np.max(degrees)}")
    print(f"  - Min degree: {np.min(degrees)}")
    
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample TempME dataset for faster training')
    parser.add_argument('--input_csv', type=str, default='ml_enron.csv',
                        help='Input CSV file path')
    parser.add_argument('--input_npy', type=str, default='ml_enron.npy',
                        help='Input edge features NPY file path')
    parser.add_argument('--input_node_npy', type=str, default='ml_enron_node.npy',
                        help='Input node features NPY file path')
    parser.add_argument('--output_prefix', type=str, default='ml_enron_sampled',
                        help='Output files prefix')
    parser.add_argument('--edge_ratio', type=float, default=0.15,
                        help='Edge sampling ratio (default: 0.15 = 15%%)')
    parser.add_argument('--temporal_windows', type=int, default=10,
                        help='Number of temporal windows for stratified sampling')
    parser.add_argument('--min_degree', type=int, default=2,
                        help='Minimum node degree to keep')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--analyze', action='store_true',
                        help='Only analyze the dataset without sampling')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.input_csv)
    else:
        sample_dataset(
            input_csv=args.input_csv,
            input_npy=args.input_npy,
            input_node_npy=args.input_node_npy,
            output_prefix=args.output_prefix,
            edge_sample_ratio=args.edge_ratio,
            temporal_windows=args.temporal_windows,
            min_node_degree=args.min_degree,
            seed=args.seed
        )
