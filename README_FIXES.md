# Implementation Summary: Geometry Reconstruction Fixes

## What Was Done

All recommended fixes have been implemented in `d:\LightMurre\main.py`:

### 1. ✅ Hyperparameter Tuning (Lines 48-55)
Reduced over-regularization and emphasized multiview constraints:
- **LR**: 1e-4 → 5e-5
- **SAMPLES_PER_VIEW**: 2048 → 1024 (higher quality)
- **WEIGHT_MV**: 1.0 → 2.0 (multiview emphasis)
- **WEIGHT_EDGE**: 0.1 → 0.02 (reduce over-smoothing)
- **WEIGHT_SURF**: 0.05 → 0.01
- **WEIGHT_MAG**: 0.05 → 0.005

### 2. ✅ Robust Loss Function (Lines 373-378)
Added Huber loss to handle outliers without warping geometry:
```python
def huber_loss(x, delta=1e-2):
    """Transition from L2 (small errors) to L1 (large errors)"""
    a = x.abs()
    return torch.where(a < delta, 0.5 * x**2, delta * (a - 0.5 * delta)).mean()
```

### 3. ✅ Magnitude Loss Fix (Line 369)
Changed denominator from `depth_pred` to `depth_init` and increased epsilon:
```python
# Before: L_mag = |δz| / (depth_pred + 1e-3)  ← unstable
# After:  L_mag = |δz| / (depth_init + 1e-2)  ← stable
```

### 4. ✅ Occlusion Masking (Lines 506-507)
Added z-buffer check to filter non-corresponding points:
```python
# Only compare points that project to consistent depth
occl_mask = (|Z_proj - Z_sampled| / Z_proj) < 0.25
# Prevents "folded geometry" from spurious matches
```

### 5. ✅ Huber Loss in Multiview (Line 519)
Applied robust loss instead of simple L2:
```python
# Before: L_mv += dist.mean()           ← sensitive to outliers
# After:  L_mv += huber_loss(dist)      ← robust
```

### 6. ✅ Diagnostic Metrics (Lines 533-542)
Added real-time monitoring:
```
Δz_mean: Average depth correction (should be non-zero)
Δz_std:  Spatial variance (should increase with training)
```

## Quick Start: Running the Fixed Training

### Option A: Resume from Last Checkpoint
```bash
cd d:\LightMurre
python main.py
```
This will load the last checkpoint and continue training with new hyperparameters.

### Option B: Fresh Start (Recommended First)
```bash
cd d:\LightMurre
rm checkpoints/deltaz_ep*.pth  # Clear old checkpoints
python main.py
```
This will start from scratch with the fixed hyperparameters.

### Option C: Quick Test (1 Epoch Only)
Edit `main.py` line 50:
```python
NUM_EPOCHS = 1  # was 40
```
Run:
```bash
python main.py
```
Watch for Δz_mean and Δz_std to be non-zero.

## What to Expect

### During Training (Every Epoch)
```
Epoch 1: loss: 0.3452  L_mv: 0.2345  Δz_mean: 0.0012  Δz_std: 0.0008
Epoch 2: loss: 0.2987  L_mv: 0.1876  Δz_mean: 0.0045  Δz_std: 0.0034
Epoch 3: loss: 0.2654  L_mv: 0.1523  Δz_mean: 0.0078  Δz_std: 0.0056
```

**Signs of success:**
- Δz_mean and Δz_std increase with epochs
- L_mv decreases steadily
- Loss curves down smoothly (not flat or oscillating)

### After Each Epoch: Evaluate Geometry
```bash
python eval.py
```
Outputs go to `checkpoints/evaluation_output/`:
```
├── before/ply/        ← Initial depth point clouds
├── after/ply/         ← Predicted depth point clouds
├── before_color/      ← Depth heatmaps (PNG)
├── after_color/       ← Depth heatmaps (PNG)
├── fused_before.ply   ← Combined initial cloud
└── fused_after.ply    ← Combined predicted cloud
```

**Open in MeshLab/CloudCompare:**
- Look for **sharpness improvement** from fused_before.ply → fused_after.ply
- Check **camera centers are outside** the cloud
- Compare **before_color/ vs after_color/** PNG heatmaps for spatial structure

## Files Created/Modified

| File | Purpose |
|------|---------|
| `main.py` | Core training script (fixes implemented) |
| `eval.py` | Evaluation and PLY export (already updated) |
| `TRAINING_FIXES.md` | Detailed explanation of each fix |
| `DIAGNOSTICS_GUIDE.md` | How to interpret metrics and debug |
| `test_training.py` | Simple test script |

## Key Metrics to Monitor

### Real-Time (During Training)
- **Δz_mean**: Should be non-zero and structured
- **Δz_std**: Should increase from ~1e-4 toward ~1e-2 to 1e-1
- **L_mv**: Should decrease (from ~0.2 toward ~0.05)
- **L_edge**: Should stay small compared to L_mv

### After Training (PLY Evaluation)
- **Before → After comparison**: Geometry sharper, fewer streaks
- **Radius distribution**: Should have more variance (not spherical)
- **Density**: Should increase (fewer holes)
- **Structure**: Should match visible features in RGB images

## If Something Goes Wrong

### Issue: Δz stays near 0
**Cause**: Regularizers still too strong
**Fix**: In main.py, reduce further:
```python
WEIGHT_EDGE = 0.005   # from 0.02
WEIGHT_SURF = 0.002   # from 0.01
WEIGHT_MAG = 0.001    # from 0.005
```

### Issue: Training diverges (NaN/Inf)
**Cause**: Learning rate too high
**Fix**: In main.py:
```python
LR = 1e-5  # from 5e-5
```

### Issue: L_mv stays constant (not decreasing)
**Cause**: Poor view overlap or occlusion mask too strict
**Fix**: In main.py around line 506:
```python
occl_mask = (torch.abs(Z_sel - sampled2) / (Z_sel + 1e-6)) < 0.5  # was 0.25
```

### Issue: Fused geometry still smooth/blob-like
**Cause**: Need more epochs
**Fix**: In main.py:
```python
NUM_EPOCHS = 80  # from 40
```

## Recommended Training Schedule

1. **Run 5 epochs** with default fixes (test stability)
2. **Evaluate** fused geometry with `eval.py`
3. **If geometry improves**: Continue training 30+ epochs
4. **If geometry stalls**: Apply one of the "If Something Goes Wrong" fixes, reset, restart

## Next Advanced Improvements (If Needed)

Once basic fixes work:
1. **SfM sparse point anchoring** (if COLMAP/MVS available)
2. **Photometric reprojection loss** (if image matches available)
3. **Multi-scale training** (coarse → fine)
4. **TSDF fusion + post-processing**

See `TRAINING_FIXES.md` section 8 for details.

## Expected Timeline

- **Epoch 1**: Model learns initial structure, Δz starts growing
- **Epochs 2-5**: Geometry refines, L_mv drops, Δz_std increases
- **Epochs 5-10**: Details emerge, artifacts reduce
- **Epochs 10+**: Fine-tuning, diminishing returns

Full 40-epoch training on CPU: ~10-20 hours
Full 40-epoch training on GPU: ~30 minutes

## Questions?

- See `DIAGNOSTICS_GUIDE.md` for metric interpretation
- See `TRAINING_FIXES.md` for technical details of each fix
- Check `main.py` comments for implementation details
