"""Extract images from partial tar, match with labels, create patient-level splits."""

import os
import sys
import tarfile
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = os.environ.get("ICH_DATA_DIR", "./data")
TAR_PATH = os.path.join(DATA_DIR, "images_partial.tar")
IMG_DIR = os.path.join(DATA_DIR, "images")
LABELS_CSV = os.path.join(DATA_DIR, "rsna_slice_labels.csv")
RESCALE_CSV = os.path.join(DATA_DIR, "rsna_rescale_values.csv")
MANIFEST_CSV = os.path.join(DATA_DIR, "manifest.csv")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")


def extract_tar():
    os.makedirs(IMG_DIR, exist_ok=True)
    print(f"Extracting from {TAR_PATH} ...")
    extracted = 0
    with tarfile.open(TAR_PATH, "r") as tar:
        for member in tar:
            try:
                tar.extract(member, IMG_DIR)
                extracted += 1
                if extracted % 5000 == 0:
                    print(f"  Extracted {extracted} files ...")
            except Exception as e:
                print(f"  Skip {member.name}: {e}")
                break
    print(f"Total extracted: {extracted}")
    return extracted


def build_manifest():
    print("Building manifest ...")
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df["rel_path"] = (
        "images/"
        + labels_df["patient_ID"]
        + "/"
        + labels_df["study_ID"]
        + "/"
        + labels_df["series_ID"]
        + "/"
        + labels_df["filename"]
    )
    label_map = {}
    for _, row in labels_df.iterrows():
        label_map[row["rel_path"]] = {
            "patient_id": row["patient_ID"],
            "any": row["any"],
            "epidural": row["epidural"],
            "intraparenchymal": row["intraparenchymal"],
            "intraventricular": row["intraventricular"],
            "subarachnoid": row["subarachnoid"],
            "subdural": row["subdural"],
        }

    records = []
    for root, dirs, files in os.walk(IMG_DIR):
        for fname in files:
            if not fname.endswith(".png"):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, DATA_DIR)
            if rel in label_map:
                info = label_map[rel]
                records.append(
                    {
                        "image_path": full,
                        "rel_path": rel,
                        "patient_id": info["patient_id"],
                        "any": info["any"],
                        "epidural": info["epidural"],
                        "intraparenchymal": info["intraparenchymal"],
                        "intraventricular": info["intraventricular"],
                        "subarachnoid": info["subarachnoid"],
                        "subdural": info["subdural"],
                    }
                )
    manifest = pd.DataFrame(records)
    manifest.to_csv(MANIFEST_CSV, index=False)
    print(
        f"Manifest: {len(manifest)} images, {manifest['patient_id'].nunique()} patients"
    )
    print(
        f"  ICH positive: {(manifest['any'] == 1).sum()} ({(manifest['any'] == 1).mean() * 100:.1f}%)"
    )
    return manifest


def create_splits(manifest):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    patient_ids = manifest["patient_id"].unique()
    train_p, temp_p = train_test_split(
        patient_ids,
        test_size=0.30,
        random_state=42,
        stratify=manifest.groupby("patient_id")["any"].max().loc[patient_ids],
    )
    val_p, test_p = train_test_split(
        temp_p,
        test_size=0.50,
        random_state=42,
        stratify=manifest.groupby("patient_id")["any"].max().loc[temp_p],
    )
    train_df = manifest[manifest["patient_id"].isin(train_p)].copy()
    val_df = manifest[manifest["patient_id"].isin(val_p)].copy()
    test_df = manifest[manifest["patient_id"].isin(test_p)].copy()
    train_df.to_csv(os.path.join(SPLIT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLIT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLIT_DIR, "test.csv"), index=False)
    print(
        f"Train: {len(train_df)} slices, {train_df['patient_id'].nunique()} patients, "
        f"{(train_df['any'] == 1).mean() * 100:.1f}% ICH+"
    )
    print(
        f"Val:   {len(val_df)} slices, {val_df['patient_id'].nunique()} patients, "
        f"{(val_df['any'] == 1).mean() * 100:.1f}% ICH+"
    )
    print(
        f"Test:  {len(test_df)} slices, {test_df['patient_id'].nunique()} patients, "
        f"{(test_df['any'] == 1).mean() * 100:.1f}% ICH+"
    )


def load_rescale_map(rescale_csv=None):
    df = pd.read_csv(rescale_csv or RESCALE_CSV)
    rescale = {}
    for _, row in df.iterrows():
        key = (row["patient_id"], row["study_id"], row["series_id"])
        rescale[key] = (float(row["rescale_slope"]), float(row["rescale_intercept"]))
    return rescale


if __name__ == "__main__":
    if not os.path.exists(MANIFEST_CSV):
        if os.path.exists(TAR_PATH):
            extract_tar()
            os.remove(TAR_PATH)
            print(f"Removed {TAR_PATH} to free disk space")
        else:
            print(f"Tar not found, using existing images in {IMG_DIR}")
        manifest = build_manifest()
    else:
        manifest = pd.read_csv(MANIFEST_CSV)
        print(f"Loaded existing manifest: {len(manifest)} images")
    create_splits(manifest)
