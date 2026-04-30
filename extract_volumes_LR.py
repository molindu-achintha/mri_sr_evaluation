import os
import sys

# TensorFlow must see this before it is imported by SynthSeg. The original
# crash is usually caused by TensorFlow finding a GPU but failing to initialise
# the CUDA/cuDNN DNN runtime in the container.
USE_GPU = os.environ.get("USE_SYNTHSEG_GPU", "0") == "1"
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
if USE_GPU and os.environ.get("SYNTHSEG_CUDNN_PATH_FIXED") != "1":
    pip_cudnn_lib = "/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib"
    if os.path.isdir(pip_cudnn_lib):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        ld_paths = [p for p in current_ld_path.split(":") if p]
        if not ld_paths or ld_paths[0] != pip_cudnn_lib:
            env = os.environ.copy()
            env["SYNTHSEG_CUDNN_PATH_FIXED"] = "1"
            env["LD_LIBRARY_PATH"] = pip_cudnn_lib + (":" + current_ld_path if current_ld_path else "")
            os.execvpe("python", ["python", *sys.argv], env)

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

# --- SETUP PATHS ---
# The structure is synseg_model (outer) -> synseg_model (inner) -> SynthSeg (package)
REPO_PATH = os.path.join(os.getcwd(), "synseg_model", "synseg_model")
sys.path.append(REPO_PATH)

# Define paths to model and labels
PATH_MODEL = os.path.join(REPO_PATH, "models", "synthseg_1.0.h5")
PATH_LABELS = os.path.join(REPO_PATH, "data", "labels_classes_priors", "synthseg_segmentation_labels.npy")

# Configure TensorFlow before importing SynthSeg's prediction module.
import tensorflow as tf

TF_THREADS = int(os.environ.get("SYNTHSEG_THREADS", "1"))
tf.config.threading.set_inter_op_parallelism_threads(TF_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(TF_THREADS)
if USE_GPU:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    tf.config.set_visible_devices([], "GPU")

# Correct Import for the official SynthSeg Repo
from SynthSeg.predict import predict

# --- CONFIGURATION ---
INPUT_DIR = "../processed/LR"
OUTPUT_DIR = "../evals-LR-LoHiResGAN"
CSV_FILENAME = "brain_volumes.csv"
SOURCE_CSV_PATH = "../evals-SR-LoHiResGAN/brain_volumes.csv"  
SAVE_SEGMENTATION = False

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
    60: "Right Ventral DC",
}


def get_voxel_volume(nii_img):
    """Calculates the volume of a single voxel in mm^3."""
    header = nii_img.header
    zooms = header.get_zooms()
    return np.prod(zooms)


def get_segmentation_path(input_path, segmentation_folder):
    filename = os.path.basename(input_path)
    if filename.endswith(".nii.gz"):
        return os.path.join(segmentation_folder, filename.replace(".nii.gz", "_synthseg.nii.gz"))
    return os.path.join(segmentation_folder, filename.replace(".nii", "_synthseg.nii"))


def extract_volumes(input_path, seg_output_path):
    filename = os.path.basename(input_path)

    # Calculate volumes from the SynthSeg segmentation.
    seg_img = nib.load(seg_output_path)
    seg_data = seg_img.get_fdata()
    voxel_vol_mm3 = get_voxel_volume(seg_img)

    unique, counts = np.unique(seg_data, return_counts=True)
    stats = dict(zip(unique.astype(int), counts))

    volumes = {"Filename": filename}

    for label_id, label_name in LABEL_MAP.items():
        count = stats.get(label_id, 0)
        volumes[label_name] = count * voxel_vol_mm3

    return volumes


def remove_last_sr_part(filename):
    """
    Remove only the final '_sr' immediately before NIfTI extension.
    Example:
      101309_T2w_inplane_ds2_sr.nii.gz -> 101309_T2w_inplane_ds2.nii.gz
    """
    if filename.endswith("_sr.nii.gz"):
        return filename[:-10] + ".nii.gz"
    if filename.endswith("_sr.nii"):
        return filename[:-7] + ".nii"
    return filename


def dedupe_keep_order(items):
    seen = set()
    unique_items = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items


def build_target_filenames(source_csv_path):
    df = pd.read_csv(source_csv_path)
    if "Filename" not in df.columns:
        raise ValueError(f"'Filename' column not found in {source_csv_path}")

    filenames = [remove_last_sr_part(name.strip()) for name in df["Filename"].dropna().astype(str)]
    return dedupe_keep_order(filenames)


def write_path_list(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    segmentation_dir = os.path.join(OUTPUT_DIR, "synthseg_segmentations")
    os.makedirs(segmentation_dir, exist_ok=True)

    target_files = build_target_filenames(SOURCE_CSV_PATH)
    if not target_files:
        print(f"No valid filenames generated from {SOURCE_CSV_PATH}")
        return

    available_files = set(os.listdir(INPUT_DIR))
    files = [f for f in target_files if f in available_files]
    missing_files = [f for f in target_files if f not in available_files]

    if missing_files:
        print(f"Warning: {len(missing_files)} files not found in {INPUT_DIR}")
        print("First few missing files:", missing_files[:10])

    if not files:
        print(f"No requested NIfTI files found in {INPUT_DIR}")
        return

    print(f"Found {len(files)} target files from CSV. Starting SynthSeg...")

    input_list_path = os.path.join(OUTPUT_DIR, "lr_synthseg_inputs.txt")
    output_list_path = os.path.join(OUTPUT_DIR, "lr_synthseg_outputs.txt")
    input_paths = [os.path.join(INPUT_DIR, f) for f in files]
    output_paths = [get_segmentation_path(path, segmentation_dir) for path in input_paths]
    write_path_list(input_list_path, input_paths)
    write_path_list(output_list_path, output_paths)

    # Run SynthSeg once so TensorFlow loads cuDNN/model state a single time.
    predict(
        path_images=input_list_path,
        path_segmentations=output_list_path,
        path_model=PATH_MODEL,
        labels_segmentation=PATH_LABELS,
        cropping=None,
        recompute=True,
    )

    all_results = []
    for f in tqdm(files):
        try:
            full_path = os.path.join(INPUT_DIR, f)
            seg_path = get_segmentation_path(full_path, segmentation_dir)
            result = extract_volumes(full_path, seg_path)
            all_results.append(result)
            if not SAVE_SEGMENTATION and os.path.exists(seg_path):
                os.remove(seg_path)
        except Exception as e:
            print(f"Error extracting volumes for {f}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        cols = ["Filename"] + [c for c in df.columns if c != "Filename"]
        df = df[cols]
        output_csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
        df.to_csv(output_csv_path, index=False)
        print(f"\nSuccess! Results saved to: {output_csv_path}")


if __name__ == "__main__":
    main()
