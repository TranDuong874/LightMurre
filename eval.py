#!/usr/bin/env python3
"""
Evaluation and export script using latest checkpoint.
"""

import os
import math
import numpy as np
from glob import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Optional for saving/merging PLYs
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

# -------------------------
# Config
# -------------------------
DATASET_ROOT = "dataset/dtu_clean"
OUT_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IN_CH = 7
OUT_CH = 1


# -------------------------
# Helper functions (from main.py)
# -------------------------
def load_image_float(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1)

def compute_ray_dirs_tensor(K, H, W, device):
    """
    K: 3x3 intrinsic matrix (numpy)
    Returns: 3,H,W tensor of normalized ray directions in camera frame
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    xs = (xs - cx) / fx
    ys = (ys - cy) / fy
    zs = np.ones_like(xs)
    
    ray_dirs = np.stack([xs, ys, zs], axis=-1)  # H,W,3
    ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=-1, keepdims=True)
    ray_dirs = torch.from_numpy(ray_dirs).permute(2, 0, 1).float()
    
    if device is not None:
        ray_dirs = ray_dirs.to(device)
    return ray_dirs

# -------------------------
# Model (small UNet-like)
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.net(x)

class SmallUNet(nn.Module):
    def __init__(self, in_ch=IN_CH, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec2 = ConvBlock(base*4, base*2)
        self.dec1 = ConvBlock(base*2, base)
        self.out = nn.Conv2d(base, OUT_CH, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.out(d1)
        return out


# -------------------------
# Dataset
# -------------------------
class DTUDataset:
    def __init__(self, root):
        self.root = root
        self.items = []
        scenes = sorted(glob(os.path.join(root, "scene_*")))
        for scene_dir in scenes:
            img_dir = os.path.join(scene_dir, "images")
            if not os.path.isdir(img_dir):
                continue
            imgs = sorted(glob(os.path.join(img_dir, "*.png")))
            for img_path in imgs:
                base = os.path.splitext(os.path.basename(img_path))[0]
                K_path = os.path.join(scene_dir, "intrinsics", f"{base}.npy")
                E_path = os.path.join(scene_dir, "extrinsics", f"{base}.npy")
                D_path = os.path.join(scene_dir, "depths", f"{base}.npy")
                if os.path.exists(K_path) and os.path.exists(E_path) and os.path.exists(D_path):
                    self.items.append({
                        "image": img_path,
                        "K": K_path,
                        "E": E_path,
                        "depth_init": D_path
                    })

# -------------------------
# Eval and export
# -------------------------
def save_ply_simple(filename, pts, cols):
    """Save point cloud to PLY file."""
    if _HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector((cols.astype(np.float32) / 255.0))
        o3d.io.write_point_cloud(filename, pcd)
    else:
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for pt, col in zip(pts, cols):
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {col[0]} {col[1]} {col[2]}\n")

def eval_and_export(model, dataset, out_ply_path):
    """
    Run model over all views, produce depth_pred per view and save various outputs.
    Organizes outputs in subfolders:
    - before/ply : individual initial depth point clouds
    - after/ply : individual predicted depth point clouds
    - before_color/ : initial depth as PNG heatmaps
    - after_color/ : predicted depth as PNG heatmaps
    - fused_before.ply : fused initial depth point cloud
    - fused_after.ply : fused predicted depth point cloud
    """
    model.eval()
    all_pts_before = []
    all_cols_before = []
    all_pts_after = []
    all_cols_after = []
    
    # Create output directory structure
    base_dir = os.path.dirname(out_ply_path)
    output_base = os.path.join(base_dir, "evaluation_output")
    before_ply_dir = os.path.join(output_base, "before", "ply")
    after_ply_dir = os.path.join(output_base, "after", "ply")
    before_color_dir = os.path.join(output_base, "before_color")
    after_color_dir = os.path.join(output_base, "after_color")
    
    for d in [before_ply_dir, after_ply_dir, before_color_dir, after_color_dir]:
        os.makedirs(d, exist_ok=True)
    
    def depth_to_colored_image(depth_map):
        """Convert depth map to colored image using viridis colormap."""
        d_min = np.nanmin(depth_map)
        d_max = np.nanmax(depth_map)
        if d_max > d_min:
            normalized = (depth_map - d_min) / (d_max - d_min)
        else:
            normalized = np.zeros_like(depth_map)
        
        # Viridis colormap
        cmap = plt.cm.viridis
        colored = cmap(normalized)
        return (colored[..., :3] * 255).astype(np.uint8)
    
    def unproject_depth_to_world(depth_map, K_np, E_np, H, W):
        """Unproject depth map to world coordinates."""
        fx = K_np[0,0]; fy = K_np[1,1]; cx = K_np[0,2]; cy = K_np[1,2]
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        Z = depth_map
        X = (xs - cx) * Z / fx
        Y = (ys - cy) * Z / fy
        pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1).reshape(-1,4)
        E_inv = np.linalg.inv(E_np)
        pts_world = (pts_cam @ E_inv.T)[:,:3]
        mask = Z.reshape(-1) > 0
        return pts_world[mask], mask
    
    with torch.no_grad():
        for it in tqdm(dataset.items, desc="Eval/export"):
            img = load_image_float(it["image"]).unsqueeze(0).to(DEVICE)  # 1,3,H,W
            depth_init = torch.from_numpy(np.load(it["depth_init"]).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)  #1,1,H,W
            K = torch.from_numpy(np.load(it["K"]).astype(np.float32)).to(DEVICE)
            E = torch.from_numpy(np.load(it["E"]).astype(np.float32)).to(DEVICE)
            H = img.shape[2]; W = img.shape[3]
            
            # Resize depth_init to match image dimensions if needed
            if depth_init.shape[2] != H or depth_init.shape[3] != W:
                depth_init = F.interpolate(depth_init, size=(H, W), mode="bilinear", align_corners=False)
            
            ray_dirs = compute_ray_dirs_tensor(K.cpu().numpy(), H, W, DEVICE).unsqueeze(0)  #1,3,H,W
            inp = torch.cat([img, depth_init, ray_dirs], dim=1)
            delta = model(inp)
            depth_pred = (depth_init + delta).clamp(min=1e-4).squeeze().cpu().numpy()  # H,W
            depth_init_np = depth_init.squeeze().cpu().numpy()  # initial depth
            rgb = (img.squeeze().permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Get scene and image names
            scene_name = os.path.basename(os.path.dirname(os.path.dirname(it["image"])))
            img_name = os.path.splitext(os.path.basename(it["image"]))[0]
            
            K_np = K.cpu().numpy()
            E_np = E.cpu().numpy()
            
            # Unproject both before and after depths
            pts_world_before, mask_before = unproject_depth_to_world(depth_init_np, K_np, E_np, H, W)
            pts_world_after, mask_after = unproject_depth_to_world(depth_pred, K_np, E_np, H, W)
            
            # RGB colors from image
            cols_rgb_before = rgb.reshape(-1,3)[mask_before]
            cols_rgb_after = rgb.reshape(-1,3)[mask_after]
            
            # BEFORE: PLY file
            before_ply = os.path.join(before_ply_dir, f"{scene_name}_{img_name}.ply")
            save_ply_simple(before_ply, pts_world_before, cols_rgb_before)
            
            # AFTER: PLY file
            after_ply = os.path.join(after_ply_dir, f"{scene_name}_{img_name}.ply")
            save_ply_simple(after_ply, pts_world_after, cols_rgb_after)
            
            # BEFORE: Color depth map PNG
            before_colored = depth_to_colored_image(depth_init_np)
            before_png = os.path.join(before_color_dir, f"{scene_name}_{img_name}.png")
            Image.fromarray(before_colored).save(before_png)
            
            # AFTER: Color depth map PNG
            after_colored = depth_to_colored_image(depth_pred)
            after_png = os.path.join(after_color_dir, f"{scene_name}_{img_name}.png")
            Image.fromarray(after_colored).save(after_png)
            
            # Accumulate for fused PLY
            all_pts_before.append(pts_world_before)
            all_cols_before.append(cols_rgb_before)
            all_pts_after.append(pts_world_after)
            all_cols_after.append(cols_rgb_after)

    # Save fused PLY files
    all_pts_before = np.concatenate(all_pts_before, axis=0)
    all_cols_before = np.concatenate(all_cols_before, axis=0)
    all_pts_after = np.concatenate(all_pts_after, axis=0)
    all_cols_after = np.concatenate(all_cols_after, axis=0)
    
    print(f"Total points in fused cloud (before): {len(all_pts_before)}")
    print(f"Total points in fused cloud (after): {len(all_pts_after)}")
    
    fused_before = os.path.join(output_base, "fused_before.ply")
    fused_after = os.path.join(output_base, "fused_after.ply")
    save_ply_simple(fused_before, all_pts_before, all_cols_before)
    save_ply_simple(fused_after, all_pts_after, all_cols_after)
    
    print(f"\nOutput structure:")
    print(f"  {output_base}/")
    print(f"  ├── before/ply/              (Initial depth PLY files)")
    print(f"  ├── after/ply/               (Predicted depth PLY files)")
    print(f"  ├── before_color/            (Initial depth heatmap PNG files)")
    print(f"  ├── after_color/             (Predicted depth heatmap PNG files)")
    print(f"  ├── fused_before.ply         (Fused initial depth point cloud)")
    print(f"  └── fused_after.ply          (Fused predicted depth point cloud)")



def main():
    # Find latest checkpoint
    ckpts = sorted(glob(os.path.join(OUT_DIR, "deltaz_ep*.pth")))
    if not ckpts:
        print(f"No checkpoints found in {OUT_DIR}")
        return
    
    latest_ckpt = ckpts[-1]
    print(f"Loading checkpoint: {latest_ckpt}")
    print(f"Using device: {DEVICE}")
    
    # Load model
    model = SmallUNet().to(DEVICE)
    ckpt = torch.load(latest_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}")
    
    # Load dataset
    dataset = DTUDataset(DATASET_ROOT)
    print(f"Dataset size: {len(dataset.items)}")
    
    # Filter to first scene only
    first_scene_items = [item for item in dataset.items if 'scene_001' in item['image']]
    print(f"First scene items: {len(first_scene_items)}")
    # Sample every Nth image for faster evaluation
    first_scene_items = first_scene_items[::5]  # Take every 5th image
    print(f"Using {len(first_scene_items)} sampled images from first scene")
    dataset.items = first_scene_items
    
    # Run evaluation
    out_ply = os.path.join(OUT_DIR, f"fused_eval_scene001.ply")
    eval_and_export(model, dataset, out_ply)
    print("Done!")

if __name__ == "__main__":
    main()
