import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import sys

# --- SETUP PATHS ---
# Add the cloned repo to Python path so we can import modules from it
# The structure is synseg_model (outer) -> synseg_model (inner) -> SynthSeg (package)
REPO_PATH = os.path.join(os.getcwd(), 'synseg_model', 'synseg_model')
sys.path.append(REPO_PATH)

# Define paths to model and labels
PATH_MODEL = os.path.join(REPO_PATH, 'models', 'synthseg_1.0.h5')
PATH_LABELS = os.path.join(REPO_PATH, 'data', 'labels_classes_priors', 'synthseg_segmentation_labels.npy')

# Correct Import for the official SynthSeg Repo
from SynthSeg.predict import predict

# --- CONFIGURATION ---
INPUT_DIR = './data/proposed/HR'
OUTPUT_DIR = './outputs'
CSV_FILENAME = 'brain_volumes.csv'
SAVE_SEGMENTATION = False  # Set to True to keep .nii files, False to delete them after extraction

# Standard FreeSurfer ColorLUT mapping for SynthSeg
LABEL_MAP = {
    0: "Background",
    2: "Left Cerebral White Matter",
    3: "Left Cerebral Cortex",
    4: "Left Lateral Ventricle",
    5: "Left Inferior Lateral Ventricle",
    7: "Left Cerebellum White Matter",
    8: "Left Cerebellum Cortex",
    10: "Left Thalamus",
    11: "Left Caudate",
    12: "Left Putamen",
    13: "Left Pallidum",
    14: "3rd Ventricle",
    15: "4th Ventricle",
    16: "Brain Stem",
    17: "Left Hippocampus",
    18: "Left Amygdala",
    26: "Left Accumbens Area",
    28: "Left Ventral DC",
    41: "Right Cerebral White Matter",
    42: "Right Cerebral Cortex",
    43: "Right Lateral Ventricle",
    44: "Right Inferior Lateral Ventricle",
    46: "Right Cerebellum White Matter",
    47: "Right Cerebellum Cortex",
    49: "Right Thalamus",
    50: "Right Caudate",
    51: "Right Putamen",
    52: "Right Pallidum",
    53: "Right Hippocampus",
    54: "Right Amygdala",
    58: "Right Accumbens Area",
    60: "Right Ventral DC"
}

def get_voxel_volume(nii_img):
    """Calculates the volume of a single voxel in mm^3."""
    header = nii_img.header
    zooms = header.get_zooms()
    return np.prod(zooms)

def process_brain_mri(input_path, output_folder):
    filename = os.path.basename(input_path)
    seg_output_path = os.path.join(output_folder, f"seg_{filename}")
    
    # 1. Run Inference (SynthSeg)
    # The predict function handles loading, inference, and saving
    predict(
        path_images=input_path,
        path_segmentations=seg_output_path,
        path_model=PATH_MODEL,
        labels_segmentation=PATH_LABELS,
        cropping=None  # Can set to e.g. [192, 192, 192] if you run out of memory
    )

    # 2. Calculate Volumes from the new segmentation
    seg_img = nib.load(seg_output_path)
    seg_data = seg_img.get_fdata()
    voxel_vol_mm3 = get_voxel_volume(seg_img)
    
    unique, counts = np.unique(seg_data, return_counts=True)
    stats = dict(zip(unique.astype(int), counts))
    
    volumes = {'Filename': filename}
    
    for label_id, label_name in LABEL_MAP.items():
        count = stats.get(label_id, 0)
        volumes[label_name] = count * voxel_vol_mm3

    # Clean up segmentation file if not needed
    if not SAVE_SEGMENTATION:
        if os.path.exists(seg_output_path):
            os.remove(seg_output_path)

    return volumes

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Filter for NIfTI files
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.nii', '.nii.gz'))]
    
    if not files:
        print(f"No NIfTI files found in {INPUT_DIR}")
        print("Please add .nii or .nii.gz files to the 'inputs' folder.")
        return

    print(f"Found {len(files)} files. Starting SynthSeg...")
    
    all_results = []
    
    for f in tqdm(files):
        try:
            full_path = os.path.join(INPUT_DIR, f)
            result = process_brain_mri(full_path, OUTPUT_DIR)
            all_results.append(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing {f}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        cols = ['Filename'] + [c for c in df.columns if c != 'Filename']
        df = df[cols]
        output_csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
        df.to_csv(output_csv_path, index=False)
        print(f"\nSuccess! Results saved to: {output_csv_path}")

if __name__ == "__main__":
    main()