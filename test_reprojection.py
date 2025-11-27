import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

SCENE_PATH = r"dataset/dtu_clean/scene_001"

IMG_DIR = os.path.join(SCENE_PATH, "images")
DEPTH_DIR = os.path.join(SCENE_PATH, "depths")
K_DIR = os.path.join(SCENE_PATH, "intrinsics")
E_DIR = os.path.join(SCENE_PATH, "extrinsics")

# ---------------------------
# Helper functions
# ---------------------------

def load_image(i):
    return np.array(Image.open(os.path.join(IMG_DIR, f"{i}.png")))

def load_depth(i):
    return np.load(os.path.join(DEPTH_DIR, f"{i}.npy"))

def project(K, Xc):
    """ Project camera-frame points to pixel coords """
    x = Xc[:,0] / Xc[:,2]
    y = Xc[:,1] / Xc[:,2]
    u = K[0,0] * x + K[0,2]
    v = K[1,1] * y + K[1,2]
    return np.stack([u, v], axis=-1)

def unproject(K, u, v, depth):
    """ Backproject pixel to 3D camera coords """
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z, 1.0])

# ---------------------------
# Pick 2 random different views
# ---------------------------

view_ids = sorted([f[:-4] for f in os.listdir(IMG_DIR) if f.endswith(".png")])
i, j = random.sample(view_ids, 2)

print(f"[INFO] Selected views: {i} → {j}")

# Load data
img_i = load_image(i)
img_j = load_image(j)

depth_i = load_depth(i)

K_i = np.load(os.path.join(K_DIR, f"{i}.npy"))
K_j = np.load(os.path.join(K_DIR, f"{j}.npy"))

E_i = np.load(os.path.join(E_DIR, f"{i}.npy"))   # world→cam
E_j = np.load(os.path.join(E_DIR, f"{j}.npy"))

E_i_inv = np.linalg.inv(E_i)
E_j     = E_j

H, W = img_i.shape[:2]

# ---------------------------
# 1. Pick random keypoint in view i
# ---------------------------

u_rand = random.randint(0, W-1)
v_rand = random.randint(0, H-1)
d = depth_i[v_rand, u_rand]

print(f"[INFO] Selected pixel: (u,v) = ({u_rand}, {v_rand}), depth={d}")

# Backproject to view-i camera frame
Xc_i = unproject(K_i, u_rand, v_rand, d)  # (4,)

# Convert camera_i → world
Xw = E_i_inv @ Xc_i   # (4,)

# Convert world → camera_j
Xc_j = E_j @ Xw       # (4,)
Xc_j = Xc_j[:3].reshape(1,3)

# Project into view j
uv_j = project(K_j, Xc_j)[0]   # [u_j, v_j]

print("[INFO] Projected into view j:", uv_j)

# ---------------------------
# Visualization
# ---------------------------

fig, ax = plt.subplots(1,2, figsize=(12,6))

ax[0].imshow(img_i)
ax[0].scatter([u_rand], [v_rand], c='red', s=80)
ax[0].set_title(f"View {i} (Original Keypoint)")

ax[1].imshow(img_j)
ax[1].scatter([uv_j[0]], [uv_j[1]], c='blue', s=80)
ax[1].set_title(f"View {j} (Projected Correspondence)")

plt.show()
