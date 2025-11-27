# ⭐ **MULTIVIEW CONSISTENCY BLUEPRINT**

### Using only:

✔ rgb.png
✔ intrinsic.npy (K)
✔ extrinsic.npy (E, world→cam)
✔ predicted depth (DepthAnything + Δz)

---

# 🟦 **0. NOTATION**

Let:

* view **i** be the source
* view **j** be the target

Inputs per-view:

```
image_i        (H, W, 3)
depth_pred_i   (H, W)
intrinsic_i    (3, 3)
extrinsic_i    (4, 4)   # world → cam
```

---

# 🟩 **1. SAMPLE A PIXEL FROM VIEW i**

Pick a random pixel in view i:

```
(u_i, v_i)
```

Get predicted depth:

```
d_i = depth_pred_i[v_i, u_i]
```

---

# 🟩 **2. LIFT PIXEL IN VIEW i → 3D POINT Xᵢ (camera space)**

Backproject using intrinsics:

[
X^{cam}_i = d_i \cdot K^{-1}_i [u_i, v_i, 1]^T
]

This gives a 3D point **in camera-i coordinates**.

---

# 🟩 **3. TRANSFORM Xᵢ TO WORLD COORDINATES**

Extrinsics are world→cam, so invert:

[
X_i = E_i^{-1} X^{cam}_i
]

This is the world-space 3D point reconstructed **from view i**.

---

# 🟩 **4. PROJECT Xᵢ INTO VIEW j → FIND PIXEL uⱼ**

Use camera j extrinsics + intrinsics:

[
X_j^{cam} = E_j X_i
]

[
u_j = K_j \frac{X^{cam}_j}{Z^{cam}_j}
]

This gives the corresponding pixel in view j:

```
(u_j, v_j)
```

If (u_j, v_j) is outside image → skip this pixel.

---

# 🟩 **5. READ DEPTH IN VIEW j AT THAT LOCATION**

Use bilinear sampling:

```
d_j = depth_pred_j(v_j, u_j)
```

---

# 🟩 **6. LIFT PIXEL IN VIEW j → 3D POINT Xⱼ (camera space)**

[
X^{cam}_j = d_j \cdot K^{-1}_j [u_j, v_j, 1]^T
]

---

# 🟩 **7. TRANSFORM Xⱼ TO WORLD COORDINATES**

[
X_j = E_j^{-1} X^{cam}_j
]

This is the world-space 3D point reconstructed **from view j**.

---

# 🟩 **8. MULTIVIEW CONSISTENCY LOSS**

[
L_{mv} = |X_i - X_j|_2
]

If depth predictions are inconsistent,
(X_i) and (X_j) are **far apart** → large loss.

Training updates Δz to reduce this.

---

# 🟩 **9. GRADIENT FLOWS TO Δz**

Because:

* (X_i) depends on depth_i → depends on Δz_i
* (X_j) depends on depth_j → depends on Δz_j

The loss backpropagates into Δz network parameters.

The model learns to:

* correct scale
* correct shape
* enforce depth consistency across views

---

# ⭐ **FINAL BLUEPRINT (compact form)**

```
for each view i:
    sample u_i,v_i
    d_i = depth_pred_i[u_i,v_i]
    X_i_cam = backproject(u_i,v_i,d_i,K_i)
    X_i = cam_to_world(E_i, X_i_cam)

    for each other view j:
        u_j, v_j = project_to_image(K_j, E_j, X_i)
        if not inside_image(u_j,v_j): continue

        d_j = depth_pred_j(u_j,v_j)    # bilinear
        X_j_cam = backproject(u_j,v_j,d_j,K_j)
        X_j = cam_to_world(E_j, X_j_cam)

        L_mv += || X_i - X_j ||₂
```

---

# ⭐ WHY THIS WORKS

Because the only possible way to minimize:

[
|X_i - X_j|
]

is for the model to adjust depth (via Δz) so that:

### → 3D reconstructed from both views matches

### → depth is consistent in world space

### → all point clouds align

### → monocular depth loses scale errors

### → depth improves without GT depth
