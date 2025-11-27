import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# ============================================================
# 0. DEPTH ANYTHING SETUP (LOAD ONCE)
# ============================================================

print("[INFO] Loading Depth Anything model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
).to(device)



# ============================================================
# 1. DTU -> MURRE CONVERSION
# ============================================================

def convert_scene(scene_path):
    """
    Converts a single scene from DTU-style layout:
        scene_xxx/
            0000/rgb.png
            0000/intrinsic.npy
            0000/extrinsic.npy
            0000/depth.npy (optional)

    To Murre layout:
        scene_xxx/
            images/0000.png
            intrinsics/0000.npy
            extrinsics/0000.npy
            depths/0000.npy
    """

    print(f"[INFO] Converting {scene_path}")

    # Make final folders
    img_out = os.path.join(scene_path, "images")
    K_out   = os.path.join(scene_path, "intrinsics")
    E_out   = os.path.join(scene_path, "extrinsics")
    D_out   = os.path.join(scene_path, "depths")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(K_out,   exist_ok=True)
    os.makedirs(E_out,   exist_ok=True)
    os.makedirs(D_out,   exist_ok=True)

    # Loop through old view folders
    for view in sorted(os.listdir(scene_path)):
        view_dir = os.path.join(scene_path, view)

        if not os.path.isdir(view_dir):
            continue
        
        if view in ["images", "intrinsics", "extrinsics", "depths"]:
            continue

        view_id = view

        rgb_path  = os.path.join(view_dir, "rgb.png")
        intr_path = os.path.join(view_dir, "intrinsic.npy")
        extr_path = os.path.join(view_dir, "extrinsic.npy")
        depth_path = os.path.join(view_dir, "depth.npy")

        # Move RGB
        if os.path.exists(rgb_path):
            shutil.move(rgb_path, os.path.join(img_out, f"{view_id}.png"))
        else:
            print(f"[WARN] Missing rgb.png in {view_dir}")

        # Move intrinsics
        if os.path.exists(intr_path):
            shutil.move(intr_path, os.path.join(K_out, f"{view_id}.npy"))
        else:
            print(f"[WARN] Missing intrinsic.npy in {view_dir}")

        # Move extrinsics
        if os.path.exists(extr_path):
            shutil.move(extr_path, os.path.join(E_out, f"{view_id}.npy"))
        else:
            print(f"[WARN] Missing extrinsic.npy in {view_dir}")

        # Move GT depth (optional)
        if os.path.exists(depth_path):
            shutil.move(depth_path, os.path.join(D_out, f"{view_id}.npy"))

        # Delete old directory
        try:
            os.rmdir(view_dir)
        except OSError:
            print(f"[WARN] Could not remove {view_dir}")


def convert_all(dataset_root):
    print("[INFO] Starting conversion...")

    scenes = sorted(os.listdir(dataset_root))

    for scene in scenes:
        scene_path = os.path.join(dataset_root, scene)
        if not os.path.isdir(scene_path):
            continue

        convert_scene(scene_path)

    print("[INFO] Conversion complete!")



# ============================================================
# 2. RUN DEPTH ANYTHING AND SAVE DEPTH MAPS
# ============================================================

def run_depthanything(scene_path):
    """
    Runs Depth Anything over all images in a scene,
    saves depth maps into:
      scene_xxx/depths_depthanything/0000.npy
    """
    print(f"[INFO] Running Depth Anything on {scene_path}")

    IMG_DIR = os.path.join(scene_path, "images")
    OUT_DEPTH_DIR = os.path.join(scene_path, "depths")
    os.makedirs(OUT_DEPTH_DIR, exist_ok=True)

    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])

    for fname in tqdm(img_files):
        view_id = os.path.splitext(fname)[0]
        img_path = os.path.join(IMG_DIR, fname)

        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Preprocess
        inputs = processor(img, return_tensors="pt").to(device)

        # Predict depth
        with torch.no_grad():
            outputs = model(**inputs)
        
        depth = outputs.predicted_depth.squeeze().cpu().numpy()  # H,W

        # Save
        np.save(os.path.join(OUT_DEPTH_DIR, f"{view_id}.npy"), depth)

    print(f"[INFO] Depth Anything complete for {scene_path}")


def run_depthanything_all(dataset_root):
    scenes = sorted(os.listdir(dataset_root))

    for scene in scenes:
        scene_path = os.path.join(dataset_root, scene)
        if not os.path.isdir(scene_path):
            continue

        # Only run if images folder exists
        if not os.path.isdir(os.path.join(scene_path, "images")):
            continue

        run_depthanything(scene_path)



# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    dataset_root = r"dataset/dtu_clean"

    # 1. Convert DTU to Murre format
    convert_all(dataset_root)

    # 2. Run Depth Anything on all scenes
    run_depthanything_all(dataset_root)

    print("[INFO] ALL DONE.")
