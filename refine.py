import os
import yaml

# Path to your data.yaml
DATA_YAML_PATH = r"D:/videos/dataset/data.yaml"

# Load class names from data.yaml
with open(DATA_YAML_PATH, 'r') as stream:
    data = yaml.safe_load(stream)
    class_names = data.get('names', [])

label_dirs = [
    r"D:/videos/dataset/labels/train",
    r"D:/videos/dataset/labels/val"
]

for label_dir in label_dirs:
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        file_path = os.path.join(label_dir, fname)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        cleaned_lines = []
        for line in lines:
            parts = line.strip().split()
            # If there was metadata appended (class_name)
            if len(parts) >= 6:
                metadata = ' '.join(parts[5:])
                if metadata in class_names:
                    idx = class_names.index(metadata)
                    # Update class id and drop metadata
                    new_line = f"{idx} {' '.join(parts[1:5])}\n"
                    cleaned_lines.append(new_line)
                else:
                    print(f"⚠️ Unknown class label '{metadata}' in {file_path}")
            elif len(parts) == 5:
                # Already cleaned (only index + bbox)
                cleaned_lines.append(line)

        # Write back cleaned labels
        with open(file_path, 'w') as f:
            f.writelines(cleaned_lines)

print("✅ Labels updated with correct class indices.")
