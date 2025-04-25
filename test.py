# for annotating the image

import os
import json
import cv2
import shutil
import random
from ultralytics import YOLO
from datetime import datetime
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────
IMAGE_DIR = r"D:/videos/output_images"
ANNOTATION_DIR = r"D:/videos/annotations"
JSON_PATH = r"D:/videos/book1.json"
DATASET_DIR = r"D:/videos/dataset"

DATE_FMT_IMAGE = "%Y%m%d_%H%M%S"
DATE_FMT_JSON = "%d-%b-%Y %H:%M:%S"
VAL_SPLIT = 0.2

model = YOLO("yolov8m.pt")
os.makedirs(ANNOTATION_DIR, exist_ok=True)

# ─── HELPERS ────────────────────────────────────────────────────────

def extract_timestamp_from_filename(filename):
    try:
        parts = filename.split('_')
        if len(parts) >= 2:
            timestamp_str = "_".join(parts[-2:]).split('.')[0]
            return datetime.strptime(timestamp_str, DATE_FMT_IMAGE)
    except Exception as e:
        print(f"Error parsing timestamp from {filename}: {e}")
    return None

def find_json_entry(timestamp, json_data):
    for entry in json_data:
        try:
            json_ts = datetime.strptime(entry["Transaction DateTime"], DATE_FMT_JSON)
            if json_ts == timestamp:
                return entry
        except:
            continue
    return None

def extract_material_suffix(material):
    if "SIZE :-" in material:
        return material.split("SIZE :-")[-1].strip().replace(" ", "").replace(";", "").replace("(", "").replace(")", "").replace(":", "").replace(",", "").replace("_", "").replace("&", "").lower()
    return "empty"

# ─── ANNOTATION GENERATION ─────────────────────────────────────────

def main():
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))]

    all_classes = set()

    for img_name in tqdm(images, desc="Annotating images"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        timestamp = extract_timestamp_from_filename(img_name)

        if not timestamp:
            print(f"❌ Could not extract timestamp: {img_name}")
            continue

        json_entry = find_json_entry(timestamp, json_data)
        if not json_entry:
            print(f"⚠️ No matching JSON entry for {img_name}")
            continue

        vehicle_type = json_entry.get("Vehicle Type", "unknown").strip()
        material = json_entry.get("Material", "").strip() or "empty"
        material_suffix = extract_material_suffix(material)

        label_class = f"{vehicle_type},{material_suffix}"
        all_classes.add(label_class)

        image = cv2.imread(img_path)
        img_h, img_w = image.shape[:2]

        results = model(img_path)[0]

        if not results.boxes or len(results.boxes) == 0:
            print(f"⚠️ No objects detected in {img_name}")
            continue

        best_box_idx = results.boxes.conf.argmax().item()
        best_box = results.boxes.xywh[best_box_idx].cpu().numpy()

        x_center, y_center, w, h = best_box
        x_center /= img_w
        y_center /= img_h
        w /= img_w
        h /= img_h

        annotation_filename = os.path.splitext(img_name)[0] + ".txt"
        annotation_path = os.path.join(ANNOTATION_DIR, annotation_filename)

        with open(annotation_path, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {label_class}\n")

        print(f"✅ Saved annotation for {img_name}")

    # ─── SPLIT DATASET ──────────────────────────────────────────────

    print("📦 Splitting dataset...")

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))]
    random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - VAL_SPLIT))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    for split, file_list in zip(["train", "val"], [train_files, val_files]):
        img_out_dir = os.path.join(DATASET_DIR, "images", split)
        lbl_out_dir = os.path.join(DATASET_DIR, "labels", split)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        for f in file_list:
            shutil.copy(os.path.join(IMAGE_DIR, f), os.path.join(img_out_dir, f))
            label_name = os.path.splitext(f)[0] + ".txt"
            src_label_path = os.path.join(ANNOTATION_DIR, label_name)
            dst_label_path = os.path.join(lbl_out_dir, label_name)
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dst_label_path)

    print("✅ Dataset split completed.")

    # ─── CREATE data.yaml ───────────────────────────────────────────

    print("📝 Creating data.yaml")

    sorted_classes = sorted(list(all_classes))
    yaml_content = f"""
path: {DATASET_DIR.replace("\\", "/")}
train: images/train
val: images/val

nc: {len(sorted_classes)}
names: {sorted_classes}
"""

    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content.strip())

    print("✅ data.yaml created.")

if __name__ == "__main__":
    main()
