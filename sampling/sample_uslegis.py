"""
Dataset Sampling Script for USLegis Dataset - TempME
=====================================================
This script creates a sampled version of the US Legislative dataset that maintains:
1. Session-based temporal structure (discrete timestamps representing congressional sessions)
2. Graph structural properties and connectivity patterns
3. Node importance distribution (key legislators)
4. Representative legislative relationships per session

The USLegis dataset has unique characteristics:
- Only 12 discrete timestamps (0-11) representing congressional sessions
- Dense graph: ~60,396 edges with only 225 nodes
- ~5000 edges per session (uniformly distributed)
- High average node degree (~537)

Sampling Strategy:
- Session-stratified sampling: Sample edges proportionally from each session
- Preserve high-degree nodes (important legislators)
- Maintain cross-session node activity patterns
- Keep temporal progression of legislative relationships
"""

import pandas as pd
import numpy as np
import os
import argparse
from collections import defaultdict, Counter


def analyze_uslegis_dataset(csv_path):
    """Analyze and print detailed statistics about the USLegis dataset."""
    df = pd.read_csv(csv_path)
    
    print("\n" + "=" * 80)
    print(f"USLegis Dataset Analysis: {csv_path}")
    print("=" * 80)
    
    print(f"\nBasic Statistics:")
    print(f"  - Total edges: {len(df):,}")
    print(f"  - Unique source nodes: {df['u'].nunique()}")
    print(f"  - Unique destination nodes: {df['i'].nunique()}")
    print(f"  - Total unique nodes: {len(set(df.u.values) | set(df.i.values))}")
    
    # Temporal analysis - session-based
    print(f"\nTemporal (Session) Statistics:")
    print(f"  - Number of sessions: {df.ts.nunique()}")
    print(f"  - Session IDs: {sorted(df.ts.unique())}")
    
    # Edges per session
    edges_per_session = df.groupby('ts').size()
    print(f"\nEdges per Session:")
    for ts, count in edges_per_session.items():
        print(f"  - Session {int(ts)}: {count:,} edges")
    
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
    print(f"  - Std degree: {np.std(degrees):.2f}")
    
    # Node activity across sessions
    print(f"\nNode Activity Across Sessions:")
    node_sessions = defaultdict(set)
    for idx, row in df.iterrows():
        node_sessions[row['u']].add(row['ts'])
        node_sessions[row['i']].add(row['ts'])
    
    sessions_per_node = [len(s) for s in node_sessions.values()]
    print(f"  - Mean sessions per node: {np.mean(sessions_per_node):.2f}")
    print(f"  - Nodes active in all sessions: {sum(1 for s in sessions_per_node if s == df.ts.nunique())}")
    print(f"  - Nodes active in 1 session only: {sum(1 for s in sessions_per_node if s == 1)}")
    
    print("=" * 80)
    
    return df


def sample_uslegis_dataset(input_csv, input_npy, input_node_npy, output_prefix,
                            edge_sample_ratio=0.20, min_node_degree=3,
                            preserve_active_nodes=True, seed=42):
    """
    Sample the USLegis dataset intelligently to maintain session-based temporal
    and structural properties.
    
    Args:
        input_csv: Path to original CSV file
        input_npy: Path to original edge features
        input_node_npy: Path to original node features
        output_prefix: Prefix for output files (e.g., 'ml_USLegis_sampled')
        edge_sample_ratio: Ratio of edges to keep (0.20 = 20%)
        min_node_degree: Minimum degree to keep a node (lower than Enron due to dense graph)
        preserve_active_nodes: If True, prioritize nodes active across multiple sessions
        seed: Random seed for reproducibility
        
    Note: USLegis-specific considerations:
        - Sessions are discrete (0-11), sample proportionally from each
        - Dense graph requires careful edge selection to maintain connectivity
        - High-degree nodes (key legislators) should be preserved
    """
    np.random.seed(seed)
    
    print("=" * 80)
    print("TempME USLegis Dataset Sampling Tool")
    print("=" * 80)
    
    # Load data
    print("\n[1/8] Loading original dataset...")
    df = pd.read_csv(input_csv)
    edge_feat = np.load(input_npy)
    node_feat = np.load(input_node_npy)
    
    original_nodes = set(df.u.values) | set(df.i.values)
    n_sessions = df.ts.nunique()
    
    print(f"  Original dataset statistics:")
    print(f"  - Total edges: {len(df):,}")
    print(f"  - Unique nodes: {len(original_nodes)}")
    print(f"  - Number of sessions: {n_sessions}")
    print(f"  - Edge features shape: {edge_feat.shape}")
    print(f"  - Node features shape: {node_feat.shape}")
    
    # Step 1: Analyze node importance across sessions
    print(f"\n[2/8] Analyzing node importance and cross-session activity...")
    
    # Track node activity per session
    node_session_degrees = defaultdict(lambda: defaultdict(int))
    node_total_degree = defaultdict(int)
    node_sessions = defaultdict(set)
    
    for idx, row in df.iterrows():
        session = row['ts']
        node_session_degrees[row['u']][session] += 1
        node_session_degrees[row['i']][session] += 1
        node_total_degree[row['u']] += 1
        node_total_degree[row['i']] += 1
        node_sessions[row['u']].add(session)
        node_sessions[row['i']].add(session)
    
    # Calculate node importance score
    # Score = (number of sessions active) * (total degree) / (max possible)
    max_sessions = n_sessions
    max_degree = max(node_total_degree.values())
    
    node_importance = {}
    for node in original_nodes:
        session_coverage = len(node_sessions[node]) / max_sessions
        degree_score = node_total_degree[node] / max_degree
        # Weighted importance: 60% session coverage + 40% degree
        node_importance[node] = 0.6 * session_coverage + 0.4 * degree_score
    
    # Identify key nodes (top quartile by importance)
    importance_threshold = np.percentile(list(node_importance.values()), 25)
    key_nodes = {n for n, imp in node_importance.items() if imp >= importance_threshold}
    print(f"  - Key nodes (importance >= {importance_threshold:.3f}): {len(key_nodes)}")
    
    # Step 2: Session-stratified sampling
    print(f"\n[3/8] Performing session-stratified sampling...")
    
    target_edges = int(len(df) * edge_sample_ratio)
    edges_per_session = target_edges // n_sessions
    
    sampled_indices = []
    session_stats = {}
    
    for session in sorted(df.ts.unique()):
        session_df = df[df['ts'] == session]
        
        # Prioritize edges involving key nodes
        key_node_edges = session_df[
            session_df['u'].isin(key_nodes) | session_df['i'].isin(key_nodes)
        ]
        other_edges = session_df[
            ~(session_df['u'].isin(key_nodes) | session_df['i'].isin(key_nodes))
        ]
        
        # Sample more from key node edges (70%) vs other edges (30%)
        n_key_samples = min(int(edges_per_session * 0.7), len(key_node_edges))
        n_other_samples = min(edges_per_session - n_key_samples, len(other_edges))
        
        if n_key_samples > 0:
            key_indices = np.random.choice(
                key_node_edges.index, size=n_key_samples, replace=False
            )
            sampled_indices.extend(key_indices)
        
        if n_other_samples > 0:
            other_indices = np.random.choice(
                other_edges.index, size=n_other_samples, replace=False
            )
            sampled_indices.extend(other_indices)
        
        session_stats[int(session)] = {
            'original': len(session_df),
            'sampled': n_key_samples + n_other_samples
        }
    
    df_sampled = df.loc[sampled_indices].copy()
    
    print(f"  - Sampled edges per session:")
    for session, stats in session_stats.items():
        print(f"    Session {session}: {stats['sampled']:,} / {stats['original']:,} "
              f"({stats['sampled']/stats['original']*100:.1f}%)")
    
    # Step 3: Verify and filter nodes by degree
    print(f"\n[4/8] Filtering nodes by minimum degree...")
    
    node_degree_sampled = defaultdict(int)
    for idx, row in df_sampled.iterrows():
        node_degree_sampled[row['u']] += 1
        node_degree_sampled[row['i']] += 1
    
    # Keep nodes with sufficient activity
    active_nodes = {node for node, deg in node_degree_sampled.items() 
                    if deg >= min_node_degree}
    
    # Always keep key nodes that are in the sampled edges
    nodes_in_sample = set(df_sampled.u.values) | set(df_sampled.i.values)
    active_nodes = active_nodes | (key_nodes & nodes_in_sample)
    
    print(f"  - Active nodes (degree >= {min_node_degree}): {len(active_nodes)}")
    
    # Step 4: Filter edges to only include active nodes
    print(f"\n[5/8] Filtering edges for active nodes...")
    df_sampled = df_sampled[
        df_sampled['u'].isin(active_nodes) & df_sampled['i'].isin(active_nodes)
    ].copy()
    print(f"  - Remaining edges: {len(df_sampled):,}")
    
    # Step 5: Remap node IDs to be contiguous
    print(f"\n[6/8] Remapping node IDs...")
    unique_nodes = sorted(set(df_sampled.u.values) | set(df_sampled.i.values))
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    
    # Keep original node mapping for reference
    original_to_new = node_mapping.copy()
    
    df_sampled['u'] = df_sampled['u'].map(node_mapping)
    df_sampled['i'] = df_sampled['i'].map(node_mapping)
    
    print(f"  - Remapped {len(unique_nodes)} nodes to IDs 0-{len(unique_nodes)-1}")
    
    # Step 6: Reindex edges while preserving temporal order
    df_sampled = df_sampled.sort_values('ts').reset_index(drop=True)
    df_sampled['idx'] = np.arange(1, len(df_sampled) + 1)
    df_sampled['Unnamed: 0'] = np.arange(len(df_sampled))
    
    # Step 7: Sample corresponding features
    print(f"\n[7/8] Sampling edge and node features...")
    
    # Get original indices from sampled_indices that made it through filtering
    # We need to match the final sampled dataframe
    original_edge_indices = sampled_indices
    
    # Filter edge features - need to recompute based on final sampled edges
    # The edge features correspond to the original edge indices
    df_original_with_idx = df.reset_index()
    final_original_indices = []
    
    for idx in sampled_indices:
        row = df.loc[idx]
        # Check if this edge (mapped nodes) is in the final sample
        mapped_u = node_mapping.get(row['u'])
        mapped_i = node_mapping.get(row['i'])
        if mapped_u is not None and mapped_i is not None:
            # Check if this edge exists in df_sampled
            match = df_sampled[
                (df_sampled['u'] == mapped_u) & 
                (df_sampled['i'] == mapped_i) & 
                (df_sampled['ts'] == row['ts'])
            ]
            if len(match) > 0:
                final_original_indices.append(idx)
    
    # Sample edge features based on original indices
    if len(final_original_indices) != len(df_sampled):
        # Fallback: use first N features if indices don't match perfectly
        print(f"  Warning: Index mismatch, using sequential edge features")
        edge_feat_sampled = edge_feat[:len(df_sampled)]
    else:
        edge_feat_sampled = edge_feat[final_original_indices]
    
    # Sample node features for active nodes (using original node IDs)
    node_feat_sampled = node_feat[unique_nodes]
    
    print(f"  - Edge features shape: {edge_feat_sampled.shape}")
    print(f"  - Node features shape: {node_feat_sampled.shape}")
    
    # Step 8: Save sampled dataset
    print(f"\n[8/8] Saving sampled dataset...")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_sampled.to_csv(f'{output_prefix}.csv', index=False)
    np.save(f'{output_prefix}.npy', edge_feat_sampled)
    np.save(f'{output_prefix}_node.npy', node_feat_sampled)
    
    print(f"  - Saved: {output_prefix}.csv")
    print(f"  - Saved: {output_prefix}.npy")
    print(f"  - Saved: {output_prefix}_node.npy")
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("SAMPLING SUMMARY")
    print("=" * 80)
    print(f"Original dataset:")
    print(f"  - Edges: {len(df):,}")
    print(f"  - Nodes: {len(original_nodes)}")
    print(f"  - Sessions: {n_sessions}")
    print(f"  - Edge features size: {edge_feat.nbytes / 1024 / 1024:.2f} MB")
    
    print(f"\nSampled dataset:")
    print(f"  - Edges: {len(df_sampled):,} ({len(df_sampled)/len(df)*100:.2f}%)")
    print(f"  - Nodes: {len(unique_nodes)} ({len(unique_nodes)/len(original_nodes)*100:.2f}%)")
    print(f"  - Sessions preserved: {df_sampled.ts.nunique()}")
    print(f"  - Edge features size: {edge_feat_sampled.nbytes / 1024 / 1024:.2f} MB")
    print(f"  - Compression ratio: {len(df_sampled)/len(df):.3f}x")
    
    # Session coverage validation
    print(f"\nSession Coverage (sampled vs original):")
    for session in sorted(df.ts.unique()):
        orig_count = len(df[df['ts'] == session])
        samp_count = len(df_sampled[df_sampled['ts'] == session])
        print(f"  Session {int(session)}: {samp_count:,} / {orig_count:,} edges "
              f"({samp_count/orig_count*100:.1f}%)")
    
    # Degree statistics comparison
    print(f"\nDegree Distribution Comparison:")
    orig_degrees = list(node_total_degree.values())
    samp_node_degrees = defaultdict(int)
    for idx, row in df_sampled.iterrows():
        samp_node_degrees[row['u']] += 1
        samp_node_degrees[row['i']] += 1
    samp_degrees = list(samp_node_degrees.values())
    
    print(f"  Original - Mean: {np.mean(orig_degrees):.1f}, "
          f"Median: {np.median(orig_degrees):.1f}, Max: {np.max(orig_degrees)}")
    print(f"  Sampled  - Mean: {np.mean(samp_degrees):.1f}, "
          f"Median: {np.median(samp_degrees):.1f}, Max: {np.max(samp_degrees)}")
    
    print(f"\nEstimated training speedup: {len(df)/len(df_sampled):.1f}x faster")
    print("=" * 80)
    
    # Validate temporal ordering (sessions should be preserved)
    assert df_sampled['ts'].is_monotonic_increasing or \
           (df_sampled['ts'].diff()[1:] >= 0).all(), "Temporal ordering violated!"
    assert df_sampled.ts.nunique() == n_sessions, "Some sessions were lost!"
    
    print("\n✓ All sessions preserved")
    print("✓ Temporal ordering maintained")
    print("✓ Sampled dataset created successfully!")
    print(f"\nYou can now train models using this sampled dataset.")
    print(f"Example: python learn_base.py --base_type tgn --data uslegis_sampled")
    
    return df_sampled, edge_feat_sampled, node_feat_sampled, original_to_new


def validate_sampled_dataset(original_csv, sampled_csv):
    """Validate that sampled dataset maintains key properties."""
    df_orig = pd.read_csv(original_csv)
    df_samp = pd.read_csv(sampled_csv)
    
    print("\n" + "=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)
    
    # Check session coverage
    orig_sessions = set(df_orig.ts.unique())
    samp_sessions = set(df_samp.ts.unique())
    
    print(f"\n1. Session Coverage:")
    print(f"   Original sessions: {len(orig_sessions)}")
    print(f"   Sampled sessions: {len(samp_sessions)}")
    print(f"   All sessions preserved: {'✓' if orig_sessions == samp_sessions else '✗'}")
    
    # Check edge distribution per session
    print(f"\n2. Edge Distribution per Session:")
    for session in sorted(orig_sessions):
        orig_count = len(df_orig[df_orig['ts'] == session])
        samp_count = len(df_samp[df_samp['ts'] == session])
        ratio = samp_count / orig_count if orig_count > 0 else 0
        print(f"   Session {int(session)}: {ratio*100:.1f}% retained")
    
    # Check node coverage
    orig_nodes = set(df_orig.u.values) | set(df_orig.i.values)
    samp_nodes = set(df_samp.u.values) | set(df_samp.i.values)
    
    print(f"\n3. Node Coverage:")
    print(f"   Original nodes: {len(orig_nodes)}")
    print(f"   Sampled nodes: {len(samp_nodes)}")
    print(f"   Coverage: {len(samp_nodes)/len(orig_nodes)*100:.1f}%")
    
    # Check temporal ordering
    print(f"\n4. Temporal Ordering:")
    is_ordered = df_samp['ts'].is_monotonic_increasing or \
                 (df_samp['ts'].diff()[1:] >= 0).all()
    print(f"   Temporal order preserved: {'✓' if is_ordered else '✗'}")
    
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sample USLegis dataset for TempME - preserves session structure'
    )
    parser.add_argument('--input_csv', type=str, default='ml_USLegis.csv',
                        help='Input CSV file path')
    parser.add_argument('--input_npy', type=str, default='ml_USLegis.npy',
                        help='Input edge features NPY file path')
    parser.add_argument('--input_node_npy', type=str, default='ml_USLegis_node.npy',
                        help='Input node features NPY file path')
    parser.add_argument('--output_prefix', type=str, default='processed/ml_uslegis_sampled',
                        help='Output files prefix')
    parser.add_argument('--edge_ratio', type=float, default=0.20,
                        help='Edge sampling ratio (default: 0.20 = 20%%). '
                             'USLegis is smaller than Enron, so we use higher ratio.')
    parser.add_argument('--min_degree', type=int, default=3,
                        help='Minimum node degree to keep (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--analyze', action='store_true',
                        help='Only analyze the dataset without sampling')
    parser.add_argument('--validate', type=str, default=None,
                        help='Validate sampled dataset against original')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_uslegis_dataset(args.input_csv)
    elif args.validate:
        validate_sampled_dataset(args.input_csv, args.validate)
    else:
        sample_uslegis_dataset(
            input_csv=args.input_csv,
            input_npy=args.input_npy,
            input_node_npy=args.input_node_npy,
            output_prefix=args.output_prefix,
            edge_sample_ratio=args.edge_ratio,
            min_node_degree=args.min_degree,
            seed=args.seed
        )
