# Training Fixes Applied

## Summary of Changes

This document describes the fixes applied to address the over-smoothing and scale collapse issues observed in the fused 3D reconstruction.

## 1. Hyperparameter Changes

**File**: `main.py` (lines 48-55)

- **LR**: 1e-4 → **5e-5** — Reduced learning rate for more stable convergence
- **SAMPLES_PER_VIEW**: 2048 → **1024** — Fewer but higher-quality constraints
- **WEIGHT_MV**: 1.0 → **2.0** — Increased multiview emphasis (was underweighted)
- **WEIGHT_EDGE**: 0.1 → **0.02** — Reduced edge regularization (was over-smoothing)
- **WEIGHT_SURF**: 0.05 → **0.01** — Reduced surface smoothness regularization
- **WEIGHT_MAG**: 0.05 → **0.005** — Reduced magnitude penalty

## 2. Loss Function Fixes

### Magnitude Loss (line 369-371)
**Problem**: Division by `depth_pred` could be unstable; dividing by the output created perverse gradients.
**Fix**: 
- Changed to divide by `depth_init` instead (the input, which is more stable)
- Increased epsilon from 1e-3 to **1e-2** for better numerical stability
- Now: `L_mag = (|δz| / (depth_init + 1e-2)).mean()`

### New Huber Loss Function (line 373-378)
**Problem**: L2 loss on reprojection errors is sensitive to outliers; warped geometry can create huge errors that dominate the loss.
**Solution**: 
```python
def huber_loss(x, delta=1e-2):
    """Huber loss for robust reprojection errors."""
    a = x.abs()
    return torch.where(a < delta, 0.5 * x**2, delta * (a - 0.5 * delta)).mean()
```
This transitions smoothly from L2 (for small errors) to L1 (for large errors), reducing outlier influence.

## 3. Occlusion Masking in Multiview Loss (lines 506-507)

**Problem**: Comparing non-corresponding points causes "folded geometry" — the model learns to satisfy constraints with geometrically impossible solutions.

**Fix**: Added z-buffer occlusion test before computing reprojection error:
```python
# Only use points that are not occluded
occl_mask = (torch.abs(Z_sel - sampled2) / (Z_sel + 1e-6)) < 0.25
pos2 = (sampled2 > 1e-5) & occl_mask
```
This ensures the projected Z-depth matches the sampled depth within 25% tolerance. Points that fail this test are likely occluded or spurious.

## 4. Robust Loss Application (line 519)

**Changed from**:
```python
L_mv += dist.mean()  # Simple L2 average
```

**Changed to**:
```python
L_mv += huber_loss(dist)  # Robust Huber loss
```

## 5. Diagnostic Metrics (lines 533-542)

Added real-time monitoring of depth predictions:
```python
delta_mean = delta_pred.detach().mean().item()
delta_std = delta_pred.detach().std().item()
pbar.set_postfix({
    ...,
    "Δz_mean": f"{delta_mean:.6f}",
    "Δz_std": f"{delta_std:.6f}"
})
```

**What to watch**:
- `Δz_mean` should be non-zero and structured (not constantly near 0)
- `Δz_std` should increase as training progresses (model learning spatially-varying corrections)
- If both stay near 0, regularizers are still dominating

## Expected Behavior After These Changes

1. **Early epochs**: L_mv should dominate; L_edge/L_surf should be modest
2. **Delta magnitude**: Should see Δz_std grow from near-0 toward 0.01–0.1 range
3. **Fused geometry**: Should transition from smooth blobs → structured shapes → better geometric detail
4. **Occlusion rejection**: Model will ignore spurious correspondences, leading to cleaner constraints

## Next Steps if Geometry Still Poor

If the fused reconstruction is still not improving:

1. **Run diagnostic script** (add to eval.py):
   ```python
   print("INIT mean,std,min,max:", depth_init.mean(), depth_init.std(), depth_init.min(), depth_init.max())
   print("PRED mean,std,min,max:", depth_pred.mean(), depth_pred.std(), depth_pred.min(), depth_pred.max())
   dz = depth_pred - depth_init
   print("DELTA mean,std,min,max:", dz.mean(), dz.std(), dz.min(), dz.max())
   ```

2. **Check extrinsic convention**: Verify E is world→cam (not cam→world)
   - Camera centers should be around the object, not inside the fused blob

3. **Consider SfM scale anchoring**: If COLMAP sparse points available, use them to anchor global scale

4. **Enable only L_mv loss**: Set all regularizer weights to 0 for 1 epoch. If geometry improves, regularizers were the bottleneck.

## Files Modified

- `d:\LightMurre\main.py`: Hyperparameters, loss functions, occlusion masking, diagnostics
