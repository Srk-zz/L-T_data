# ─── INSTALL YOLOv8 ───────────────────────────────────────────────────────
!pip install ultralytics --upgrade -q

# Optional: check that GPU is visible
!nvidia-smi

# ─── IMPORTS ─────────────────────────────────────────────────────────────
import yaml
from ultralytics import YOLO

# ─── USER CONFIG ─────────────────────────────────────────────────────────
# The original YAML shipped in your input dataset:
ORIG_YAML   = "/kaggle/input/d/srivathsan1510/dataset-1/dataset/data.yaml"

# The root folder that actually contains `images/train` & `images/val`
DATA_ROOT   = "/kaggle/input/d/srivathsan1510/dataset-1/dataset"

# Where to write the corrected YAML (must be writable)
WORK_YAML   = "/kaggle/working/updated_data.yaml"

# Which backbone size to start from
MODEL_SIZE  = "yolov8m.pt"   # or 'yolov8s.pt' / 'yolov8n.pt'

# ─── PATCH YOUR YAML ─────────────────────────────────────────────────────
with open(ORIG_YAML, 'r') as f:
    data_cfg = yaml.safe_load(f)

# override the path to point at the Kaggle‐mounted folder
data_cfg['path'] = DATA_ROOT

# write out a new YAML for YOLO
with open(WORK_YAML, 'w') as f:
    yaml.safe_dump(data_cfg, f)

print(f"✅ Updated YAML written to {WORK_YAML}")
print("Contents now:")
print(yaml.safe_dump(data_cfg, sort_keys=False))

# ─── TRAIN ────────────────────────────────────────────────────────────────
model = YOLO(MODEL_SIZE)
model.train(
    data=WORK_YAML,      # our patched YAML
    epochs=50,
    imgsz=640,
    batch=16,
    name="custom_yolov8_vehicle",
    project="/kaggle/working"
)

# Best weights → /kaggle/working/custom_yolov8_vehicle/weights/best.pt
print("✅ Training complete.")
