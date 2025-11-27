#!/usr/bin/env python3
"""
Minimal Δz training + eval script.

Usage:
    python train_deltaz.py

Dataset layout (example):
dataset_root/
  dtu_clean/
    scene_001/
      images/0000.png
      intrinsics/0000.npy
      extrinsics/0000.npy
      depths/0000.npy   # DepthAnything prediction (initial)
"""

import os
import math
import random
from glob import glob
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

# Optional for saving/merging PLYs
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

# -------------------------
# Config
# -------------------------
DATASET_ROOT = "dataset/dtu_clean"  # parent folder that contains scene_xxx folders
OUT_DIR = "checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_FREQ = 10

# Training hyperparams (fixed for better geometry learning)
NUM_EPOCHS = 40
LR = 5e-5                  # reduced from 1e-4
BATCH_SIZE = 4             # number of views per batch (all from same scene due to sampler)
SAMPLES_PER_VIEW = 1024    # reduced from 2048 for higher quality constraints
WEIGHT_MV = 2.0            # increased from 1.0 - emphasize multiview
WEIGHT_EDGE = 0.02         # reduced from 0.1
WEIGHT_SURF = 0.01         # reduced from 0.05
WEIGHT_MAG = 0.005         # reduced from 0.05

# Model input channels: RGB (3) + depth_init (1) + ray_dirs (3) = 7
IN_CH = 7
OUT_CH = 1  # delta_z

# -------------------------
# Utilities
# -------------------------
def load_image_float(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    # to torch C,H,W
    return torch.from_numpy(arr).permute(2,0,1)

def load_npy(path):
    return np.load(path)

def save_ply_xyzrgb(filename, pts, cols):
    # pts: N x 3 (numpy), cols: N x 3 uint8
    if _HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector((cols.astype(np.float32) / 255.0))
        o3d.io.write_point_cloud(filename, pcd)
    else:
        with open(filename, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write("element vertex %d\n" % pts.shape[0])
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p,c in zip(pts, cols):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

# -------------------------
# Dataset
# -------------------------
class MurreDataset(Dataset):
    """
    items: list of dict:
      {scene_id, view_id, image_path, depth_init_path, K_path, E_path}
    """
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.scenes = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
        self.items = []
        for s in self.scenes:
            sp = os.path.join(dataset_root, s)
            img_dir = os.path.join(sp, "images")
            depth_dir = os.path.join(sp, "depths")
            K_dir = os.path.join(sp, "intrinsics")
            E_dir = os.path.join(sp, "extrinsics")
            if not os.path.isdir(img_dir):
                continue
            imgs = sorted(glob(os.path.join(img_dir, "*.png")))
            for imgp in imgs:
                vid = Path(imgp).stem
                item = {
                    "scene": s,
                    "view": vid,
                    "image": imgp,
                    "depth_init": os.path.join(depth_dir, f"{vid}.npy"),
                    "K": os.path.join(K_dir, f"{vid}.npy"),
                    "E": os.path.join(E_dir, f"{vid}.npy"),
                }
                self.items.append(item)

        # group by scene for SceneBatchSampler
        self.groups = {}
        for idx, it in enumerate(self.items):
            self.groups.setdefault(it["scene"], []).append(idx)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        # we just return paths so collate_fn can control loading
        return it

class SceneBatchSampler(Sampler):
    """
    yields lists of indices, each list of length batch_size and from same scene.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.scenes = list(self.dataset.groups.keys())

    def __iter__(self):
        # shuffle scenes
        scenes = self.scenes.copy()
        random.shuffle(scenes)
        for s in scenes:
            idxs = self.dataset.groups[s]
            if len(idxs) < self.batch_size:
                # if not enough views, sample with replacement
                selected = [random.choice(idxs) for _ in range(self.batch_size)]
            else:
                selected = random.sample(idxs, self.batch_size)
            yield selected

    def __len__(self):
        return len(self.scenes)

def compute_ray_dirs_tensor(K, H, W, device):
    """
    returns [3,H,W] ray directions in camera coordinates (unit vectors)
    such that for a pixel (u,v): direction = K^{-1} [u,v,1]
    """
    fx = float(K[0,0]); fy = float(K[1,1]); cx = float(K[0,2]); cy = float(K[1,2])
    xs = torch.arange(0, W, device=device).view(1, -1).expand(H, W)
    ys = torch.arange(0, H, device=device).view(-1, 1).expand(H, W)
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    dirs = torch.stack([x, y, torch.ones_like(x)], dim=0)  # 3,H,W
    return dirs  # not normalized; ray scaling is handled by depth multiplication

def collate_scene(batch_items):
    """
    batch_items is list of dicts from dataset.__getitem__. 
    We will load images, depth_init, K, E and return a dict with tensors:
      images: [B,3,H,W]
      depth_init: [B,1,H,W]
      K: [B,3,3]
      E: [B,4,4]   (world->cam)
      ray_dirs: [B,3,H,W]
      view_ids, scene
    """
    device = DEVICE
    B = len(batch_items)
    images = []
    depth_inits = []
    Ks = []
    Es = []
    ray_dirs = []
    H = W = None

    for it in batch_items:
        img_t = load_image_float(it["image"])
        d_init = np.load(it["depth_init"]).astype(np.float32)
        K = np.load(it["K"]).astype(np.float32)
        E = np.load(it["E"]).astype(np.float32)
        # ensure shapes
        if H is None:
            H = int(img_t.shape[1]); W = int(img_t.shape[2])
        
        # if image has different res, resize to match first image
        if img_t.shape[1] != H or img_t.shape[2] != W:
            img_t = F.interpolate(img_t.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
        
        # if depth has different res, resize to match image
        if d_init.shape != (H, W):
            t = torch.from_numpy(d_init).unsqueeze(0).unsqueeze(0).float()
            t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
            d_init = t.squeeze().numpy()

        images.append(img_t)
        depth_inits.append(torch.from_numpy(d_init).unsqueeze(0))  # 1,H,W
        Ks.append(torch.from_numpy(K))
        Es.append(torch.from_numpy(E))

    images = torch.stack(images)        # B,3,H,W
    depth_inits = torch.stack(depth_inits)  # B,1,H,W
    Ks = torch.stack(Ks)  # B,3,3
    Es = torch.stack(Es)  # B,4,4

    # compute ray dirs per view (camera coords)
    ray_dirs_batch = []
    for b in range(B):
        rd = compute_ray_dirs_tensor(Ks[b].numpy(), H, W, None)  # 3,H,W (CPU)
        ray_dirs_batch.append(rd)
    ray_dirs = torch.stack(ray_dirs_batch)  # B,3,H,W

    out = {
        "images": images,
        "depth_init": depth_inits,
        "K": Ks,
        "E": Es,
        "ray_dirs": ray_dirs,
        "meta": batch_items
    }
    return out

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
# Geometry helpers (torch)
# -------------------------
def unproject_pixel_to_cam(u, v, depth, K):
    # u,v can be tensors (N,), depth (N,), K: [3,3] or [B,3,3]
    fx = K[...,0,0]; fy = K[...,1,1]; cx = K[...,0,2]; cy = K[...,1,2]
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return torch.stack([X, Y, Z], dim=-1)  # (...,3)

def cam_to_world(points_cam, E):
    # points_cam: (...,3), E: [4,4] world->cam
    # Convert to homogeneous then world = E_inv @ [Xc,1]
    B = E.shape[0]
    if points_cam.dim() == 2 and B == 1:
        pts4 = torch.cat([points_cam, torch.ones(points_cam.shape[0],1, device=points_cam.device)], dim=1)
        E_inv = torch.inverse(E)
        w = (E_inv @ pts4.T).T[..., :3]
        return w
    # General vectorized implementation for batches where needed elsewhere
    # We'll build ad-hoc batch usage in training code.
    raise NotImplementedError("Use batched path in training loop.")

def project_world_to_image(points_world, K, E):
    # points_world: (N,3)
    # E: world->cam [4,4], K: [3,3]
    # returns u,v and z_cam
    N = points_world.shape[0]
    pts4 = torch.cat([points_world, torch.ones(N,1, device=points_world.device)], dim=1)  # N,4
    pts_cam = (E @ pts4.T).T
    Xc = pts_cam[:,0]; Yc = pts_cam[:,1]; Zc = pts_cam[:,2].clamp(min=1e-6)
    u = K[0,0] * (Xc / Zc) + K[0,2]
    v = K[1,1] * (Yc / Zc) + K[1,2]
    return torch.stack([u,v,Zc], dim=1)

# Bilinear sample depth at fractional coords using grid_sample
def bilinear_sample_depth(depth_map, uv):
    """
    depth_map: [B,1,H,W]
    uv: [M,2] in pixels (u,v) absolute
    returns depths: [M]
    """
    B,_,H,W = depth_map.shape
    # create grid in -1..1 coords; uv are absolute pixels
    u = uv[:,0]; v = uv[:,1]
    # normalize
    x = (u / (W-1)) * 2.0 - 1.0
    y = (v / (H-1)) * 2.0 - 1.0
    grid = torch.stack([x,y], dim=1).view(1, -1, 2)  # 1, M, 2
    grid = grid.unsqueeze(0)  # 1,1,M,2 not needed for grid_sample; reshape below
    # grid_sample expects [N,H_out,W_out,2], so we provide as [1,M,1,2] and sample via permute
    grid_rs = grid.view(1, -1, 1, 2)
    sampled = F.grid_sample(depth_map, grid_rs, mode='bilinear', padding_mode='zeros', align_corners=True)
    # sampled: [B,1,M,1]
    sampled = sampled.view(B,1,-1).squeeze(0).squeeze(0)
    return sampled  # size M

# Faster approach: for our multiview sampling, we'll build custom samplers inside the train loop
# -------------------------
# Loss definitions
# -------------------------
def edge_aware_loss(depth, image):
    # depth: [B,1,H,W], image: [B,3,H,W]
    def grad_x(t): return t[..., :, 1:] - t[..., :, :-1]
    def grad_y(t): return t[..., 1:, :] - t[..., :-1, :]

    dx = grad_x(depth)
    dy = grad_y(depth)
    Ix = torch.mean(torch.abs(grad_x(image)), dim=1, keepdim=True)
    Iy = torch.mean(torch.abs(grad_y(image)), dim=1, keepdim=True)
    wx = torch.exp(-Ix)
    wy = torch.exp(-Iy)
    L = (torch.abs(dx) * wx).mean() + (torch.abs(dy) * wy).mean()
    return L

def surface_smoothness_loss(depth):
    def grad_x(t): return t[..., :, 1:] - t[..., :, :-1]
    def grad_y(t): return t[..., 1:, :] - t[..., :-1, :]
    dx = grad_x(depth)
    dy = grad_y(depth)
    dxx = grad_x(dx)
    dyy = grad_y(dy)
    return (dxx.abs().mean() + dyy.abs().mean())

def magnitude_loss(delta_z, depth_init, eps=1e-2):
    # Fixed: divide by depth_init instead of depth_pred, use larger eps
    return (torch.abs(delta_z) / (depth_init + eps)).mean()

def huber_loss(x, delta=1e-2):
    """Huber loss for robust reprojection errors."""
    a = x.abs()
    return torch.where(a < delta, 0.5 * x**2, delta * (a - 0.5 * delta)).mean()

# -------------------------
# Training & Eval
# -------------------------
def train_epoch(model, opt, loader, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch["images"].to(DEVICE)      # B,3,H,W
        depth_init = batch["depth_init"].to(DEVICE)  # B,1,H,W
        Ks = batch["K"].to(DEVICE)
        Es = batch["E"].to(DEVICE)
        ray_dirs = batch["ray_dirs"].to(DEVICE)
        meta = batch["meta"]
        B,_,H,W = images.shape

        # Build model input: concat [image, depth_init, ray_dirs]
        inp = torch.cat([images, depth_init, ray_dirs], dim=1)  # B,7,H,W

        delta_pred = model(inp)  # B,1,H,W (can be negative)
        depth_pred = (depth_init + delta_pred).clamp(min=1e-4)  # B,1,H,W

        # 1) Edge-aware + surface + magnitude
        L_edge = edge_aware_loss(depth_pred, images)
        L_surf = surface_smoothness_loss(depth_pred)
        L_mag = magnitude_loss(delta_pred, depth_init)

        # 2) Multiview consistency with occlusion masking and Huber loss
        # We'll sample SAMPLES_PER_VIEW random pixels per view, unproject -> world -> project into other views and compare
        L_mv = 0.0
        cnt = 0
        for b in range(B):
            # sample pixels
            # random uniform sampling but ignore border ~2 pixels
            us = torch.randint(2, W-2, (SAMPLES_PER_VIEW,), device=DEVICE).float()
            vs = torch.randint(2, H-2, (SAMPLES_PER_VIEW,), device=DEVICE).float()

            # depths at those pixels
            di = F.grid_sample(depth_pred[b:b+1], torch.stack([
                (us/(W-1))*2.0-1.0, (vs/(H-1))*2.0-1.0], dim=1).view(1,SAMPLES_PER_VIEW,1,2),
                mode='bilinear', padding_mode='zeros', align_corners=True).view(-1)  # S

            # mask out near-zero
            valid_mask = di > 1e-5
            if valid_mask.sum() < 32:
                continue
            us = us[valid_mask]; vs = vs[valid_mask]; di = di[valid_mask]

            # unproject to camera coords
            Kb = Ks[b]
            Eb = Es[b]
            # compute X_i_cam: (N,3)
            Xc = unproject_pixel_to_cam(us, vs, di, Kb)  # (N,3)
            # to homogeneous camera coords:
            ones = torch.ones(Xc.shape[0],1, device=DEVICE)
            Xc4 = torch.cat([Xc, ones], dim=1)  # N,4
            # cam->world using E_inv
            E_inv = torch.inverse(Eb.to(DEVICE))
            Xw = (E_inv @ Xc4.T).T[:, :3]  # N,3

            # project into all other views in batch (we'll sample up to K other views)
            for bj in range(B):
                if bj == b: continue
                Kj = Ks[bj]; Ej = Es[bj]
                # project world -> cam j
                pts4 = torch.cat([Xw, torch.ones(Xw.shape[0],1, device=DEVICE)], dim=1)
                pts_camj = (Ej.to(DEVICE) @ pts4.T).T  # N,4
                Xcj = pts_camj[:, :3]
                Zcj = Xcj[:, 2].clone()
                # valid only if in front
                in_front = Zcj > 1e-5
                if in_front.sum() < 8:
                    continue
                Xcj = Xcj[in_front]; Zcj = Zcj[in_front]
                # compute uv coords
                u = (Kj[0,0] * (Xcj[:,0] / Zcj) + Kj[0,2])
                v = (Kj[1,1] * (Xcj[:,1] / Zcj) + Kj[1,2])
                # mask in image bounds
                inside = (u >= 0) & (u <= (W-1)) & (v >= 0) & (v <= (H-1))
                if inside.sum() < 8:
                    continue
                u = u[inside]; v = v[inside]
                # sample depth_pred in view j at these uv (bilinear)
                # build normalized grid for grid_sample
                xnorm = (u / (W-1)) * 2.0 - 1.0
                ynorm = (v / (H-1)) * 2.0 - 1.0
                grid = torch.stack([xnorm, ynorm], dim=1).view(1, -1, 1, 2)  # 1,M,1,2
                sampled = F.grid_sample(depth_pred[bj:bj+1], grid, mode='bilinear', padding_mode='zeros', align_corners=True)
                dj = sampled.view(-1)
                # filter dj positive
                pos = dj > 1e-5
                if pos.sum() < 8:
                    continue
                dj = dj[pos]
                u = u[pos]; v = v[pos]

                # unproject in camera j -> world
                Xcj_cam = unproject_pixel_to_cam(u, v, dj, Kj)
                Xcj4 = torch.cat([Xcj_cam, torch.ones(Xcj_cam.shape[0],1, device=DEVICE)], dim=1)
                Xwj = (torch.inverse(Ej.to(DEVICE)) @ Xcj4.T).T[:, :3]

                # Recompute consistent pipeline to get matching Xw subset indices with occlusion mask:
                pts4_all = torch.cat([Xw, torch.ones(Xw.shape[0],1, device=DEVICE)], dim=1)
                pts_camj_all = (Ej.to(DEVICE) @ pts4_all.T).T
                Xcamj_all = pts_camj_all[:, :3]
                Z_all = Xcamj_all[:,2]
                in_front_all = Z_all > 1e-5
                u_all = (Kj[0,0] * (Xcamj_all[:,0] / Z_all) + Kj[0,2])
                v_all = (Kj[1,1] * (Xcamj_all[:,1] / Z_all) + Kj[1,2])
                inside_all = (u_all >= 0) & (u_all <= (W-1)) & (v_all >= 0) & (v_all <= (H-1))
                valid_all = in_front_all & inside_all
                idxs = valid_all.nonzero(as_tuple=False).view(-1)
                if idxs.numel() == 0:
                    continue
                # sample depths at u_all[idxs], v_all[idxs]
                u_sel = u_all[idxs]; v_sel = v_all[idxs]
                xnorm_sel = (u_sel / (W-1)) * 2.0 - 1.0
                ynorm_sel = (v_sel / (H-1)) * 2.0 - 1.0
                grid_sel = torch.stack([xnorm_sel, ynorm_sel], dim=1).view(1, -1, 1, 2)
                sampled2 = F.grid_sample(depth_pred[bj:bj+1], grid_sel, mode='bilinear', padding_mode='zeros', align_corners=True).view(-1)
                Z_sel = Z_all[idxs]
                
                # OCCLUSION MASK: only use points that are not occluded
                # Check if projected Z matches sampled depth (within 25% tolerance)
                occl_mask = (torch.abs(Z_sel - sampled2) / (Z_sel + 1e-6)) < 0.25
                
                pos2 = (sampled2 > 1e-5) & occl_mask
                if pos2.sum() < 8:
                    continue
                sampled2 = sampled2[pos2]
                idxs2 = idxs[pos2]

                # Now build X_w_i_selected and X_w_j from sampled2 for Huber loss
                Xw_sel = Xw[idxs2]            # M,3
                # unproject using sampled2 to world
                Xcj_cam_sel = unproject_pixel_to_cam(u_sel[pos2], v_sel[pos2], sampled2, Kj)
                Xcj4_sel = torch.cat([Xcj_cam_sel, torch.ones(Xcj_cam_sel.shape[0],1, device=DEVICE)], dim=1)
                Xwj_sel = (torch.inverse(Ej.to(DEVICE)) @ Xcj4_sel.T).T[:, :3]

                # Compute robust Huber loss
                dist = torch.norm(Xw_sel - Xwj_sel, dim=1)
                L_mv += huber_loss(dist)
                cnt += 1

        if cnt > 0:
            L_mv = L_mv / max(1, cnt)
        else:
            L_mv = torch.tensor(0.0, device=DEVICE)

        loss = WEIGHT_MV * L_mv + WEIGHT_EDGE * L_edge + WEIGHT_SURF * L_surf + WEIGHT_MAG * L_mag

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += float(loss.item())
        if (batch_idx % PRINT_FREQ) == 0:
            # Diagnostic info
            delta_stats = delta_pred.detach()
            delta_mean = delta_stats.mean().item()
            delta_std = delta_stats.std().item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "L_mv": f"{L_mv.item():.4f}",
                "L_edge": f"{L_edge.item():.4f}",
                "Δz_mean": f"{delta_mean:.6f}",
                "Δz_std": f"{delta_std:.6f}"
            })

    return total_loss / (len(loader) + 1e-12)
def eval_and_export(model, dataset, out_ply_path):
    """
    Run model over all views, produce depth_pred per view and fuse into one PLY by transforming each
    unprojected point cloud into world coordinates (using E_inv).
    """
    model.eval()
    all_pts = []
    all_cols = []
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
            rgb = (img.squeeze().permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)

            # unproject to world
            fx = K[0,0].item(); fy = K[1,1].item(); cx = K[0,2].item(); cy = K[1,2].item()
            ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            Z = depth_pred
            X = (xs - cx) * Z / fx
            Y = (ys - cy) * Z / fy
            pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1).reshape(-1,4)
            E_inv = np.linalg.inv(E.cpu().numpy())
            pts_world = (pts_cam @ E_inv.T)[:,:3]
            mask = Z.reshape(-1) > 0
            pts_world = pts_world[mask]
            cols = rgb.reshape(-1,3)[mask]
            all_pts.append(pts_world)
            all_cols.append(cols)

    if len(all_pts) == 0:
        print("No points exported.")
        return
    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    # Optional subsample if huge
    if pts.shape[0] > 5_000_000:
        idx = np.random.choice(pts.shape[0], 5_000_000, replace=False)
        pts = pts[idx]; cols = cols[idx]
    save_ply_xyzrgb(out_ply_path, pts, cols)
    print("Saved fused PLY:", out_ply_path)

# -------------------------
# Main
# -------------------------
def main():
    random.seed(42)
    torch.manual_seed(42)

    dataset = MurreDataset(DATASET_ROOT)
    sampler = SceneBatchSampler(dataset, batch_size=BATCH_SIZE)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_scene, num_workers=4, pin_memory=True)

    model = SmallUNet(in_ch=IN_CH).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = 1e9
    for ep in range(1, NUM_EPOCHS+1):
        avg = train_epoch(model, opt, loader, ep)
        print(f"Epoch {ep} avg loss: {avg:.4f}")
        # save checkpoint
        ckpt = os.path.join(OUT_DIR, f"deltaz_ep{ep:03d}.pth")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep}, ckpt)

        # quick eval/export every few epochs
        if ep % 10 == 0:
            out_ply = os.path.join(OUT_DIR, f"fused_ep{ep:03d}.ply")
            eval_and_export(model, dataset, out_ply)

    print("Training finished.")

if __name__ == "__main__":
    main()
