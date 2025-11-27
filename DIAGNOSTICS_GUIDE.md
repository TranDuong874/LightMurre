# Diagnostics & Interpretation Guide

After running training with the fixes, use these guides to evaluate whether the fixes are working.

## 1. Real-Time Training Diagnostics

When you run `python main.py`, watch the progress bar for these metrics:

### Key Metrics to Monitor

```
Epoch 1: [========>  ] Δz_mean: 0.001234  Δz_std: 0.000567  L_mv: 0.2345  loss: 0.3456
                      ↑                    ↑                  ↑             ↑
                      mean depth change    std of changes     multiview      total loss
```

### What Each Metric Means

| Metric | Good Value | Problem |
|--------|-----------|---------|
| `Δz_mean` | Non-zero, ~0.001-0.05 | If near 0: regularizers dominating; if huge: unstable |
| `Δz_std` | Increasing with epochs; 0.0005-0.1 | If constant near 0: model not learning; if huge: exploding |
| `L_mv` | Decreasing → toward 0.001-0.05 | If increasing: diverging; if stuck: bad constraints |
| `L_edge` | Decreasing, much smaller than L_mv | Should be minimal, not dominant |
| `loss` | Steady decrease per epoch | If plateaus quickly: learning rate too high/low |

### Quick Interpretation

**Scenario A: Good Training**
```
Epoch 1: Δz_mean: 0.0012  Δz_std: 0.0008  L_mv: 0.245
Epoch 2: Δz_mean: 0.0045  Δz_std: 0.0034  L_mv: 0.198
Epoch 3: Δz_mean: 0.0078  Δz_std: 0.0056  L_mv: 0.156
```
→ All increasing steadily. Model learning structured corrections. **Expect good geometry!**

**Scenario B: Over-regularization (old problem)**
```
Epoch 1: Δz_mean: 0.00001  Δz_std: 0.00001  L_mv: 0.234
Epoch 2: Δz_mean: 0.00001  Δz_std: 0.00001  L_mv: 0.233
Epoch 3: Δz_mean: 0.00001  Δz_std: 0.00001  L_mv: 0.232
```
→ Δz frozen near zero. Regularizers still too strong. **Reduce WEIGHT_EDGE, WEIGHT_SURF further.**

**Scenario C: Unstable Learning**
```
Epoch 1: Δz_mean: 0.0012   Δz_std: 0.0008  L_mv: 0.245  loss: 0.250
Epoch 2: Δz_mean: NaN or Inf
```
→ Exploding gradients. **Reduce LR, or check for division-by-zero bugs.**

**Scenario D: No Constraints**
```
Epoch 1: L_mv: 0.0  (always zero)
```
→ Multiview loss never computed (sample count filtering too strict). **Check batch size / dataset overlap.**

---

## 2. Post-Training Evaluation

After each epoch, look at checkpoint evaluation:

### Save a Diagnostic Depth Image

Add this to `eval.py` after computing `depth_pred`:

```python
# Save delta heatmap
delta = depth_pred - depth_init_np
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.subplot(131); plt.imshow(depth_init_np, cmap='viridis'); plt.title('Initial Depth')
plt.subplot(132); plt.imshow(depth_pred, cmap='viridis'); plt.title('Predicted Depth')
plt.subplot(133); plt.imshow(delta, cmap='seismic'); plt.title('Delta (Δz)')
plt.colorbar(); plt.tight_layout()
plt.savefig(f"checkpoints/depth_compare_ep{epoch}.png")
plt.close()
```

### Interpret Depth Visualizations

**Good signs:**
- `delta` heatmap shows **structured, spatially-varying** corrections (not uniform)
- `delta` values **not near zero everywhere** (model is making changes)
- **High-texture areas** in image → larger corrections (edge-aware learning working)
- **Smooth areas** → small corrections (regularizers working)

**Bad signs:**
- `delta` is **pure white/blue** (all zero or all one value) → stuck solution
- `delta` **has huge outlier pixels** → needs robust loss (this should be fixed now)
- `predicted_depth` looks **nearly identical to initial_depth** → model not learning

---

## 3. Fused 3D Reconstruction Evaluation

### Visual Inspection

In your 3D viewer, look at:

```
checkpoints/evaluation_output/
├── before_color/     ← Initial depth heatmaps (PNG)
├── after_color/      ← Predicted depth heatmaps (PNG)
├── fused_before.ply  ← Initial fused cloud
└── fused_after.ply   ← Predicted fused cloud
```

### Metrics for Before → After Comparison

**Expected improvements:**

| Metric | Before (Bad) | After (Good) |
|--------|-------------|------------|
| Shape | Smooth convex blob | Sharp, structured geometry |
| Streaks | Many long rays from camera | Rare, mostly resolved |
| Density | Thin, sparse | Dense point cloud |
| Symmetry | Radially biased | Realistic asymmetry |
| Camera centers | Inside/near cloud | Clearly exterior to cloud |

### Point Cloud Statistics

Compute in viewer or script:

```python
import numpy as np

# Load fused PLY
pts_before = np.load("before_fused.npy")  # your points
pts_after = np.load("after_fused.npy")

# Radial distance statistics
r_before = np.linalg.norm(pts_before, axis=1)
r_after = np.linalg.norm(pts_after, axis=1)

print(f"Before: radius mean={r_before.mean():.2f}, std={r_before.std():.2f}")
print(f"After:  radius mean={r_after.mean():.2f}, std={r_after.std():.2f}")

# If std is much larger after → good (geometry less biased toward single scale)
```

---

## 4. Failure Mode Diagnosis

If geometry still bad after fixes:

### Check A: Extrinsics Sign

```python
# In training/eval script
E = np.load("dataset/dtu_clean/scene_001/extrinsics/0000.npy")
E_inv = np.linalg.inv(E)
cam_center = (E_inv @ np.array([0,0,0,1]))[:3]
print("Camera center (world coords):", cam_center)

# Plot this point in your fused PLY viewer
# Q: Is it clearly OUTSIDE the object?
#    YES → extrinsics OK
#    NO  → might be inverted; try Ej = torch.inverse(Ej) in code
```

### Check B: Depth Unit Mismatch

```python
# Sample one image
K = np.load("dataset/dtu_clean/scene_001/intrinsics/0000.npy")
depth = np.load("dataset/dtu_clean/scene_001/depths/0000.npy")

print("K diagonal (fx, fy):", K[0,0], K[1,1])
print("Image size (approx):", depth.shape)
# Typical: K ~[2000, 2000], image ~1920x1440 → OK
# Mismatch: K ~[500, 500] for 1920x1440 → depth units might be in decimeters!

print("Depth range:", depth.min(), depth.max())
# Typical: 1-10 meters for indoor scenes
# If 100-1000: possibly decimeters or different unit
```

### Check C: Regularizer Ablation (1 epoch test)

Temporarily set in `main.py`:
```python
WEIGHT_EDGE = 0.0
WEIGHT_SURF = 0.0
WEIGHT_MAG = 0.0
# Only L_mv
```
Run 1 epoch. If geometry improves dramatically → regularizers still too strong.

---

## 5. Tuning Checklist

If fixes help but geometry still imperfect:

- [ ] Run 5+ epochs (not just 1)
- [ ] Reduce WEIGHT_EDGE further (try 0.005)
- [ ] Increase WEIGHT_MV (try 3.0)
- [ ] Lower LR more (try 1e-5) for stability
- [ ] Verify occlusion mask tolerance (25% default) isn't too strict
- [ ] Check batch sampling: are all views overlapping? Try smaller BATCH_SIZE
- [ ] Visualize `delta` images per-epoch to confirm learning progression

---

## 6. Expected Training Curve

### Loss Over Time (Good Case)
```
Epoch 1:  loss=0.352  L_mv=0.245  Δz_mean=0.0012
Epoch 2:  loss=0.298  L_mv=0.198  Δz_mean=0.0045
Epoch 3:  loss=0.267  L_mv=0.156  Δz_mean=0.0078
Epoch 4:  loss=0.245  L_mv=0.127  Δz_mean=0.0112
Epoch 5:  loss=0.228  L_mv=0.105  Δz_mean=0.0145
```
→ Smooth, steady decrease. Good convergence.

### Loss Over Time (Bad Case - Still Over-regularized)
```
Epoch 1:  loss=0.352  L_mv=0.245  Δz_mean=0.00001
Epoch 2:  loss=0.351  L_mv=0.244  Δz_mean=0.00001
Epoch 3:  loss=0.350  L_mv=0.243  Δz_mean=0.00001
```
→ Flat. Regularizers dominating. **Action: Reduce WEIGHT_EDGE, WEIGHT_SURF, WEIGHT_MAG by 10x.**

---

## Questions to Ask Yourself

1. **Is Δz_std increasing over epochs?**
   - YES → Model learning. Good sign.
   - NO → Either learning rate too low OR regularizers too strong.

2. **Is L_mv decreasing steadily?**
   - YES → Multiview constraints working. Good sign.
   - NO → Either bad feature matches OR occlusion mask too strict.

3. **Is fused geometry becoming sharper?**
   - YES → Corrections are meaningful.
   - NO → Corrections are noise or biased (scale collapse, radial bias).

4. **Are camera centers clearly outside the fused cloud?**
   - YES → Extrinsics likely correct.
   - NO → Transform might be inverted or scene mis-scaled.

---

## Summary

**The fixes address:**
1. **Over-regularization** ← Reduced regularizer weights
2. **Outlier sensitivity** ← Added Huber loss
3. **False correspondences** ← Added occlusion masking
4. **Unstable gradients** ← Improved magnitude loss formulation
5. **Poor exploration** ← Increased multiview emphasis

**Monitor these to confirm fixes work:**
- Δz_std increasing (model making structured changes)
- L_mv decreasing (constraints being satisfied)
- Fused geometry sharpening (corrections meaningful, not noise)

If any of these don't happen, use the failure mode diagnostics above!
