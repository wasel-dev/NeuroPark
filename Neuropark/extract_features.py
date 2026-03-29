import os
import nibabel as nib
import numpy as np
import pandas as pd

# -------------------------
# Paths
# -------------------------
scan_folder = "data_scans"
mask_folder = "segmentation"
output_csv = "etv_unity_dataset.csv"

dataset = []

# -------------------------
# Loop through MRI scan files
# -------------------------
for scan_file in os.listdir(scan_folder):
    if scan_file.endswith(".nii") or scan_file.endswith(".nii.gz"):
        patient_id = scan_file.split(".")[0]
        scan_path = os.path.join(scan_folder, scan_file)
        mask_path = os.path.join(mask_folder, f"{patient_id}_mask.nii.gz")
        
        if not os.path.exists(mask_path):
            print(f"Mask not found for {patient_id}, skipping...")
            continue

        # Load MRI scan (optional, can be used to derive synthetic values)
        img = nib.load(scan_path)
        data = img.get_fdata()
        voxel_volume = np.prod(img.header.get_zooms())
        
        # Load segmentation mask
        mask = nib.load(mask_path)
        mask_data = mask.get_fdata()

        # -------------------------
        # Feature extraction (synthetic / Unity-style)
        # -------------------------
        # Relative position (px, py, pz)
        px = np.random.uniform(-10, 10)
        py = np.random.uniform(-10, 10)
        pz = np.random.uniform(-10, 10)

        # Relative rotation (rx, ry, rz)
        rx = np.random.uniform(-180, 180)
        ry = np.random.uniform(-180, 180)
        rz = np.random.uniform(-180, 180)

        # Depth along B-forward axis
        depth = np.random.uniform(0, 20)

        # Target label: success (0=fail, 1=success) - can be synthetic or from clinical data
        success = np.random.randint(0, 2)

        # Append row to dataset
        dataset.append([px, py, pz, rx, ry, rz, depth, success])

# -------------------------
# Save to CSV
# -------------------------
columns = ["px", "py", "pz", "rx", "ry", "rz", "depth", "success"]
df = pd.DataFrame(dataset, columns=columns)
df.to_csv(output_csv, index=False)
print(f"Unity-style CSV saved to {output_csv}")