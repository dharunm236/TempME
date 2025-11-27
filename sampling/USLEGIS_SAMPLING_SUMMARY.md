# TempME USLegis Dataset Sampling - Summary

## Overview

This document describes the intelligent sampling methodology for the US Legislative (USLegis) dataset, which has unique characteristics compared to the Enron dataset.

## Dataset Characteristics

### USLegis vs Enron Comparison

| Property              | USLegis           | Enron               |
| --------------------- | ----------------- | ------------------- |
| **Total Edges**       | 60,396            | 125,235             |
| **Total Nodes**       | 225               | 184                 |
| **Unique Timestamps** | 12 (sessions)     | 22,632 (continuous) |
| **Timestamp Type**    | Discrete sessions | Continuous time     |
| **Mean Node Degree**  | 536.85            | ~68                 |
| **Edge Feature Dim**  | 1                 | 32                  |
| **Node Feature Dim**  | 172               | 32                  |

### Key USLegis Characteristics

1. **Session-Based Structure**: 12 congressional sessions (0-11), not continuous time
2. **Dense Graph**: Very high average node degree (~537)
3. **Uniform Distribution**: ~5,000 edges per session
4. **High Connectivity**: Almost all nodes connected across multiple sessions

## Sampling Strategy

### Why Different Approach Than Enron?

The Enron dataset has continuous timestamps, requiring **temporal stratified sampling** across time windows. USLegis has discrete sessions, requiring **session-stratified sampling** to preserve:

1. **Session Integrity**: Each of 12 congressional sessions must be represented
2. **Legislative Patterns**: Cross-session relationships between legislators
3. **Node Importance**: Key legislators (high-degree, multi-session active) preserved

### Sampling Algorithm

1. **Node Importance Scoring**

   ```
   importance = 0.6 * (sessions_active / total_sessions) + 0.4 * (degree / max_degree)
   ```

   - Prioritizes legislators active across multiple sessions
   - Considers total legislative activity (degree)

2. **Session-Stratified Sampling**

   - Sample proportionally from each of 12 sessions
   - 70% of samples from edges involving key nodes
   - 30% from other edges (maintains graph diversity)

3. **Minimum Degree Filtering**
   - Keep nodes with degree >= 3 (lower than Enron due to density)
   - Always preserve key nodes even with lower degree

## Sampling Results

### Original Dataset

- **Edges**: 60,396
- **Nodes**: 225
- **Sessions**: 12
- **Edge Features**: 0.46 MB

### Sampled Dataset (20% ratio)

- **Edges**: 8,832 (14.62%)
- **Nodes**: 224 (99.56%)
- **Sessions**: 12 (100%)
- **Edge Features**: 0.07 MB

### Session Coverage

| Session | Original | Sampled | Retention |
| ------- | -------- | ------- | --------- |
| 0       | 5,050    | 795     | 15.7%     |
| 1       | 5,050    | 732     | 14.5%     |
| 2       | 5,048    | 704     | 13.9%     |
| 3       | 5,050    | 704     | 13.9%     |
| 4       | 4,950    | 704     | 14.2%     |
| 5       | 5,142    | 707     | 13.7%     |
| 6       | 5,050    | 710     | 14.1%     |
| 7       | 5,151    | 705     | 13.7%     |
| 8       | 4,950    | 704     | 14.2%     |
| 9       | 5,151    | 704     | 13.7%     |
| 10      | 4,854    | 749     | 15.4%     |
| 11      | 4,950    | 914     | 18.5%     |

### Degree Distribution Change

| Metric            | Original | Sampled |
| ----------------- | -------- | ------- |
| **Mean Degree**   | 536.9    | 78.9    |
| **Median Degree** | 498.0    | 66.5    |
| **Max Degree**    | 1198     | 185     |

## Usage

### Generate Sampled Dataset

```bash
# Default: 20% sampling (recommended for USLegis)
python sample_uslegis.py --edge_ratio 0.20

# Faster: 15% sampling
python sample_uslegis.py --edge_ratio 0.15 --output_prefix processed/ml_uslegis_sampled_15

# Analyze original dataset
python sample_uslegis.py --analyze

# Validate sampled dataset
python sample_uslegis.py --validate processed/ml_uslegis_sampled.csv
```

### Train Models

```bash
# Train TGN on sampled dataset
python learn_base.py --base_type tgn --data uslegis_sampled --n_epoch 50

# Train explainer
python temp_exp_main.py --base_type tgn --data uslegis_sampled --n_epoch 40

# Verify enhancements
python enhance_main.py --data uslegis_sampled --base_type tgn
```

## What's Preserved

### ✅ Preserved Properties

1. **All 12 Sessions**: Every congressional session has edges
2. **Session Ratios**: Similar edge distribution per session (~14-18%)
3. **Key Legislators**: High-importance nodes prioritized in sampling
4. **Cross-Session Patterns**: Nodes active across sessions retained
5. **Temporal Ordering**: Session sequence maintained

### ⚡ Changed Properties

1. **Edge Count**: Reduced from 60,396 to 8,832
2. **Node Degrees**: Mean degree reduced from 537 to 79
3. **Graph Density**: Lower density after sampling

## Files Generated

### Output Files

- `processed/ml_uslegis_sampled.csv` - Sampled edge list
- `processed/ml_uslegis_sampled.npy` - Sampled edge features
- `processed/ml_uslegis_sampled_node.npy` - Sampled node features

### Modified Configuration Files

The following files support `uslegis_sampled`:

- `learn_base.py` - Base model training
- `temp_exp_main.py` - Explainer training
- `enhance_main.py` - Enhancement verification
- `processed/data_preprocess.py` - Data preprocessing
- `utils/null_model.py` - Null model utilities
- `processed/utils/null_model.py` - Processed utilities
- `visualize_explanations.py` - Visualization tools

## Estimated Training Time

| Configuration     | Estimated Time per Epoch | 50 Epochs          |
| ----------------- | ------------------------ | ------------------ |
| **Full USLegis**  | ~20-30 min               | 17-25 hours        |
| **Sampled (20%)** | ~3-5 min                 | **2.5-4 hours** ✅ |
| **Sampled (15%)** | ~2-4 min                 | **1.5-3 hours** ✅ |

**Training speedup: ~6.8x faster**

## Research Validity

### For Publication

The sampling methodology ensures research validity by:

1. **Preserving Session Structure**: Legislative patterns maintained
2. **Keeping Key Actors**: Important legislators prioritized
3. **Maintaining Temporal Dynamics**: Session-to-session evolution preserved
4. **Statistical Significance**: Sufficient edges per session for analysis

### Recommendations

1. Use sampled dataset for hyperparameter tuning
2. Report sampling methodology in experimental section
3. Compare relative improvements between models
4. Validate key findings on full dataset if time permits

## Technical Details

### Node Importance Formula

```python
importance = 0.6 * (sessions_active / max_sessions) + 0.4 * (node_degree / max_degree)
```

### Sampling Priority

- **Key nodes**: Top 75% by importance score
- **Edge selection**: 70% from key node edges, 30% others

### Reproducibility

- Random seed: 42 (default)
- Deterministic sampling with `np.random.seed(42)`
