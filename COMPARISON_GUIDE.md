# TensorBoard Logging & Explainer Comparison Guide

## Why TensorBoard Logs Only Appeared in `enhance_main.py`

**Answer:** The original `temp_exp_main.py` (explainer training script) **did NOT have TensorBoard logging implemented**, while `enhance_main.py` (verification script) already had complete TensorBoard integration.

### What I Fixed:
✅ Added TensorBoard support to `temp_exp_main.py`:
- Imported `SummaryWriter` from `torch.utils.tensorboard`
- Added `init_tensorboard()` function to create timestamped log directories
- Modified `train()` function to accept `tb_writer` parameter
- Modified `eval_one_epoch()` to accept and use `tb_writer`
- Added logging for all training metrics (Loss, APS, AUC, Acc, Fidelity metrics)
- Added logging for all test metrics including Ratio metrics

### TensorBoard Log Naming Convention:
- **Explainer Training** (`temp_exp_main.py`): `{base_type}_{data}_{timestamp}_explainer`
  - Example: `tgn_enron_sampled_20231110_143000_explainer`
  
- **Verification/Enhancement** (`enhance_main.py`): `{base_type}_{data}_{timestamp}`
  - Example: `tgn_enron_sampled_20231110_143000`

---

## How to Compare Optimized vs Original Explainer

### Option 1: Compare Explainer Training Performance (RECOMMENDED)

**What to Compare:** Run `temp_exp_main.py` on BOTH versions and compare training metrics

#### Steps:

1. **Train with ORIGINAL explainer (current version):**
   ```powershell
   python temp_exp_main.py --base_type tgn --data enron_sampled --n_epoch 50
   ```
   - TensorBoard log: `tensorboard_logs/tgn_enron_sampled_{timestamp}_explainer/`
   - This trains the explainer from scratch

2. **Switch to OPTIMIZED explainer version:**
   - Restore the `models/explainer.py` with optimizations (guided walks, dependency-aware sampling)
   - Add back the optimization parameters to `__init__()`:
     - `use_temporal_guidance=True`
     - `use_dependency_aware_sampling=True`

3. **Train with OPTIMIZED explainer:**
   ```powershell
   python temp_exp_main.py --base_type tgn --data enron_sampled --n_epoch 50
   ```
   - TensorBoard log: `tensorboard_logs/tgn_enron_sampled_{timestamp2}_explainer/`

4. **Compare in TensorBoard:**
   ```powershell
   tensorboard --logdir=tensorboard_logs
   ```

#### Key Metrics to Compare:

**Training Metrics:**
- `Train/Loss` - Lower is better
- `Train/Aps` - Average Precision Score (higher is better)
- `Train/Auc` - Area Under Curve (higher is better)
- `Train/Acc` - Accuracy (higher is better)
- `Train/Fidelity_Prob` - How well explanations match model predictions (higher is better)
- `Train/Fidelity_Logit` - Fidelity based on logits (higher is better)

**Test Metrics:**
- `Test/Loss` - Validation loss
- `Test/Aps`, `Test/Auc`, `Test/Acc` - Generalization performance
- `Test/Ratio_APS_AUC` - **Most important metric** (ratio of explanation quality)
- `Test/Ratio_AUC`, `Test/Ratio_ACC` - Other ratio metrics

**What to Look For:**
- ✅ Better convergence speed (fewer epochs to reach good performance)
- ✅ Higher final accuracy/APS/AUC scores
- ✅ Better fidelity scores (explanations match model predictions)
- ✅ Higher ratio metrics (explanation quality)

---

### Option 2: Compare Verification Performance

**What to Compare:** Run `enhance_main.py` using saved explainer models from both versions

This is LESS useful because:
- ❌ It only tests the explainer once it's trained, not the training process
- ❌ You need to train explainers separately first
- ❌ The base model (TGN) is the same in both cases

**Only use this if you want to verify:**
- Whether the optimized explainer produces better explanations on the same data
- Whether the explanations lead to better model predictions when used as guidance

---

## Recommended Comparison Strategy for Research Paper

### For Publication-Quality Comparison:

1. **Run 3-5 independent trials** of each explainer version:
   ```powershell
   # Original explainer (repeat 3-5 times)
   python temp_exp_main.py --base_type tgn --data enron_sampled --n_epoch 100
   
   # Optimized explainer (repeat 3-5 times)
   python temp_exp_main.py --base_type tgn --data enron_sampled --n_epoch 100
   ```

2. **Track these metrics:**
   - Training time per epoch
   - Convergence rate (epochs to reach 95% of best performance)
   - Final test metrics (APS, AUC, Acc)
   - Explanation quality (Fidelity scores)
   - Stability across runs (standard deviation)

3. **Statistical Comparison:**
   - Mean ± Std for each metric
   - T-test or Wilcoxon test for significance
   - Learning curves showing convergence

4. **Visualizations for Paper:**
   - Learning curves (Test/Aps vs epochs)
   - Box plots comparing final metrics
   - Confusion matrices
   - Example explanations from both versions

---

## Current Status

✅ **COMPLETED:**
- TensorBoard logging added to `temp_exp_main.py`
- Both scripts now log the same types of metrics
- Log directories have clear naming conventions
- Original explainer code restored and working

📝 **READY TO RUN:**
```powershell
# Start training with current (original) explainer
python temp_exp_main.py --base_type tgn --data enron_sampled --n_epoch 50

# View logs in real-time
tensorboard --logdir=tensorboard_logs
```

🔍 **TO COMPARE WITH OPTIMIZATIONS:**
1. Save current results
2. Restore optimized explainer code (I can help with this)
3. Run training again with same parameters
4. Compare TensorBoard logs side-by-side

---

## TensorBoard Commands

```powershell
# View all logs
tensorboard --logdir=tensorboard_logs

# View specific run
tensorboard --logdir=tensorboard_logs/tgn_enron_sampled_{timestamp}_explainer

# Access TensorBoard
# Open browser to: http://localhost:6006
```

---

## Summary Answer to Your Questions

**Q1: Why do I get tensorboard logs only executing the enhance_main file?**
- Because the original `temp_exp_main.py` didn't have TensorBoard logging implemented
- I've now added it - both files will create TensorBoard logs

**Q2: Which should I be comparing?**
- **Compare the explainer TRAINING logs from `temp_exp_main.py`** (not enhance_main)
- Run `temp_exp_main.py` with original explainer → get TensorBoard log
- Run `temp_exp_main.py` with optimized explainer → get TensorBoard log  
- Compare these two logs to see which explainer trains better
- The `enhance_main.py` logs are less useful for comparison as they just verify a trained explainer

**Q3: Should I compare tensorboard logs from enhance_main on both versions?**
- ❌ No, that's not the right comparison
- ✅ Compare `temp_exp_main.py` logs from both explainer versions
- The enhance_main just tests if the trained explainer works with the base model
