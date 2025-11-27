# TempME: Towards the Explainability of Temporal Graph Neural Networks via Motif Discovery

This is the Pytorch Implementation of [**_TempME:Towards the Explainability of Temporal Graph Neural Networks via Motif Discovery_**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5c5bc3553815adb4d1a8a5b8701e41a9-Abstract-Conference.html) [[arXiv]](https://arxiv.org/abs/2310.19324)

## Quick Start for Fast Training

### Create a Sampled Dataset (Recommended for Fast Experimentation)

For rapid experimentation and research publication, we provide a dataset sampling tool that creates a smaller version of the Enron dataset while preserving temporal and structural properties. This reduces training time from 30-40 minutes per epoch to approximately 5-7 minutes per epoch.

```bash
# Create a sampled dataset (15% of original edges, ~6.7x faster training)
python sample_dataset.py --edge_ratio 0.15 --temporal_windows 10 --min_degree 2

# For even faster training (10% of edges, ~10x faster)
python sample_dataset.py --edge_ratio 0.10 --output_prefix ml_enron_sampled_10

# To analyze dataset statistics
python sample_dataset.py --analyze
```

**Sampling Parameters:**
- `--edge_ratio`: Fraction of edges to keep (default: 0.15 = 15%)
- `--temporal_windows`: Number of temporal windows for stratified sampling (default: 10)
- `--min_degree`: Minimum node degree to retain (default: 2)
- `--output_prefix`: Output file prefix (default: ml_enron_sampled)

The sampled dataset preserves:
- ✅ Temporal ordering and patterns
- ✅ Graph structural properties
- ✅ Node importance distribution
- ✅ Research validity for publication

### Train a Base Model

To start, you'll need to train a base model. Our framework supports several base model types, including TGAT, TGN, and GraphMixer. To train your model, use the following command, replacing `${type}` with your chosen base model type (e.g., `tgat`, `tgn`, `graphmixer`) and `${dataset}` with the name of your dataset.

**Using the sampled dataset (recommended for fast training):**
```bash
python learn_base.py --base_type ${type} --data enron_sampled
```

**Using the full dataset:**
```bash
python learn_base.py --base_type ${type} --data ${dataset}
```

**Example commands:**
```bash
# Train TGN on sampled Enron dataset (fast)
python learn_base.py --base_type tgn --data enron_sampled --n_epoch 50

# Train TGAT on sampled dataset
python learn_base.py --base_type tgat --data enron_sampled --n_epoch 50

# Train GraphMixer on sampled dataset
python learn_base.py --base_type graphmixer --data enron_sampled --n_epoch 50
```


### Train an Explainer
Once you have a base model, the next step is to train an explainer. Use the following command to train your explainer:

**Using the sampled dataset:**
```bash
python temp_exp_main.py --base_type ${type} --data enron_sampled
```

**Using the full dataset:**
```bash
python temp_exp_main.py --base_type ${type} --data ${dataset}
```

### Verify Enhancement Effect
To evaluate the effectiveness of the explanatory subgraphs extracted by the explainer, use the following command:

**Using the sampled dataset:**
```bash
python enhance_main.py --data enron_sampled --base_type ${type}
```

**Using the full dataset:**
```bash
python enhance_main.py --data ${dataset} --base_type ${type}
```

### Automated Batch Training (Recommended)

For convenience, use the automated batch training script to train all models:

```bash
# Train all models (TGN, TGAT, GraphMixer) sequentially
python batch_train.py --mode sequential --dataset enron_sampled --base_epochs 50 --explainer_epochs 40

# Train in two phases: all base models first, then all explainers
python batch_train.py --mode two-phase --dataset enron_sampled --base_epochs 50 --explainer_epochs 40

# Train a single model with full pipeline
python batch_train.py --mode single --model tgn --dataset enron_sampled --base_epochs 50 --explainer_epochs 40
```

The batch training script provides:
- ✅ Automatic execution of full training pipeline
- ✅ Progress tracking with time estimates
- ✅ Comprehensive summary of all training tasks
- ✅ Error handling and reporting

### Verify Your Setup

Before starting training, verify your environment:

```bash
python test_setup.py
```

This will check:
- Dataset files are present and valid
- Configuration files are updated
- Python dependencies are installed
- Project structure is correct

## Training Time Comparison

| Dataset | Edges | Training Time per Epoch | Total Time (50 epochs) |
|---------|-------|------------------------|------------------------|
| Full Enron | 125,235 | 30-40 min | 25-33 hours |
| Sampled Enron (15%) | 18,780 | 5-7 min | 4-6 hours ✅ |
| Sampled Enron (10%) | ~12,500 | 3-5 min | 2.5-4 hours ✅ |

**Note:** The sampled dataset is ideal for rapid experimentation, hyperparameter tuning, and research publication. It maintains the essential temporal and structural properties of the original dataset.

## Additional Resources

- **FAST_TRAINING_GUIDE.md** - Comprehensive guide for 12-hour training pipeline
- **SAMPLING_SUMMARY.md** - Technical details about the sampling methodology
- **test_setup.py** - Automated environment verification
- **batch_train.py** - Automated batch training pipeline

## Citation
If you find this work useful, please consider citing:

```
@article{chen2024tempme,
  title={Tempme: Towards the explainability of temporal graph neural networks via motif discovery},
  author={Chen, Jialin and Ying, Rex},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```