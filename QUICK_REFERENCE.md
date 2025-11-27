# TempME Fast Training - Quick Reference Card

## 🚀 One-Command Setup

```bash
# 1. Sample dataset (1 minute)
python sample_dataset.py --edge_ratio 0.15

# 2. Verify setup (30 seconds)
python test_setup.py

# 3. Train all models automatically (12-18 hours)
python batch_train.py --mode sequential --dataset enron_sampled
```

## 📊 Dataset Options

| Sample Rate | Edges | Training Speed | Use Case |
|-------------|-------|----------------|----------|
| 15% (default) | 18,780 | 6.7x faster | Balanced, recommended |
| 10% | ~12,500 | 10x faster | Very fast experiments |
| 20% | ~25,000 | 5x faster | More data, slower |

```bash
# Create different samples
python sample_dataset.py --edge_ratio 0.10 --output_prefix ml_enron_10pct
python sample_dataset.py --edge_ratio 0.15 --output_prefix ml_enron_sampled  # default
python sample_dataset.py --edge_ratio 0.20 --output_prefix ml_enron_20pct
```

## 🎯 Training Modes

### Single Model (4-7 hours)
```bash
# Train TGN
python batch_train.py --mode single --model tgn --dataset enron_sampled

# Train TGAT  
python batch_train.py --mode single --model tgat --dataset enron_sampled

# Train GraphMixer
python batch_train.py --mode single --model graphmixer --dataset enron_sampled
```

### All Models Sequential (12-18 hours)
```bash
python batch_train.py --mode sequential --dataset enron_sampled --base_epochs 50 --explainer_epochs 40
```

### All Models Two-Phase (12-18 hours)
```bash
# Trains all base models first, then all explainers
python batch_train.py --mode two-phase --dataset enron_sampled --base_epochs 50 --explainer_epochs 40
```

## 🔧 Manual Training (Step by Step)

```bash
# Step 1: Train base model
python learn_base.py --base_type tgn --data enron_sampled --n_epoch 50

# Step 2: Train explainer
python temp_exp_main.py --base_type tgn --data enron_sampled --n_epoch 40

# Step 3: Verify enhancement
python enhance_main.py --data enron_sampled --base_type tgn
```

## ⚡ Speed Optimization Tips

### Faster Training
```bash
# Reduce epochs (still gets good results)
--base_epochs 40 --explainer_epochs 30

# Use 10% sample
--dataset enron_10pct

# Increase batch size (if GPU allows)
python learn_base.py --base_type tgn --data enron_sampled --bs 1024
```

### Memory Optimization
```bash
# Reduce batch size
--bs 256

# Reduce neighbor sampling
--n_degree 20
```

## 📈 Expected Timeline

| Task | 15% Sample | 10% Sample |
|------|------------|------------|
| TGN Base (50 epochs) | 4-6 hours | 2.5-4 hours |
| TGN Explainer (40 epochs) | 3-4 hours | 2-3 hours |
| TGAT Base (50 epochs) | 4-6 hours | 2.5-4 hours |
| TGAT Explainer (40 epochs) | 3-4 hours | 2-3 hours |
| GraphMixer Base (50 epochs) | 4-6 hours | 2.5-4 hours |
| GraphMixer Explainer (40 epochs) | 3-4 hours | 2-3 hours |
| **Total (all 3 models)** | **12-18 hours** | **7-12 hours** |

## 🎓 Research Paper Workflow

```bash
# 1. Quick experimentation (10% sample)
python sample_dataset.py --edge_ratio 0.10 --output_prefix ml_enron_10pct
python batch_train.py --mode sequential --dataset enron_10pct --base_epochs 30

# 2. Full results (15% sample)
python sample_dataset.py --edge_ratio 0.15
python batch_train.py --mode sequential --dataset enron_sampled --base_epochs 50

# 3. Final validation (optional - full dataset)
python batch_train.py --mode sequential --dataset enron --base_epochs 50
```

## 🔍 Troubleshooting

### Setup Issues
```bash
python test_setup.py  # Run diagnostics
```

### Out of Memory
```bash
python learn_base.py --base_type tgn --data enron_sampled --bs 256
```

### Check GPU
```bash
nvidia-smi  # Monitor GPU usage
```

### Training Too Slow
```bash
# Use smaller sample
python sample_dataset.py --edge_ratio 0.10

# Or reduce epochs
python batch_train.py --mode sequential --base_epochs 30 --explainer_epochs 25
```

## 📁 Output Files

### After Sampling
- `ml_enron_sampled.csv` - Sampled edges
- `ml_enron_sampled.npy` - Edge features
- `ml_enron_sampled_node.npy` - Node features

### After Training
- `params/tgnn/tgn_enron_sampled.pt` - TGN base model
- `params/explainer/tgn/enron_sampled.pt` - TGN explainer
- `tensorboard_logs/` - Training logs

### View Logs
```bash
tensorboard --logdir=tensorboard_logs
```

## 🎯 Common Use Cases

### Quick Test (2-3 hours)
```bash
python sample_dataset.py --edge_ratio 0.10
python batch_train.py --mode single --model tgn --dataset enron_10pct --base_epochs 30 --explainer_epochs 25
```

### Full Benchmark (12-18 hours)
```bash
python sample_dataset.py --edge_ratio 0.15
python batch_train.py --mode sequential --dataset enron_sampled --base_epochs 50 --explainer_epochs 40
```

### Paper Submission (if time allows)
```bash
python batch_train.py --mode sequential --dataset enron --base_epochs 100 --explainer_epochs 80
```

## 📚 Documentation

- `README.md` - Getting started
- `FAST_TRAINING_GUIDE.md` - Detailed guide
- `SAMPLING_SUMMARY.md` - Technical details
- `test_setup.py` - Environment verification
- `batch_train.py` - Automated training

## ✅ Pre-Flight Checklist

- [ ] Python 3.8+ installed
- [ ] PyTorch with CUDA support
- [ ] GPU available (check with `nvidia-smi`)
- [ ] 10GB+ free disk space
- [ ] Run `python test_setup.py` - all tests pass
- [ ] Sampled dataset created
- [ ] Ready to train! 🚀

## 💡 Pro Tips

1. **Start with 10% sample** for initial experiments
2. **Use batch_train.py** for hands-off training
3. **Monitor with TensorBoard** to track progress
4. **Train during off-hours** (overnight/weekend)
5. **Save checkpoints** regularly
6. **Test one model first** before running all
7. **Use two-phase mode** if memory constrained

---

**Need Help?** 
- Run: `python test_setup.py`
- Check: `FAST_TRAINING_GUIDE.md`
- Verify GPU: `nvidia-smi`
