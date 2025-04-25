import os

# Define the exact class names from your data.yaml (must match exactly)
class_names = [
    'Dumper,20mmdown',
    'Dumper,5mmdowndust',
    'Dumper,75mmdown',
    'Dumper,empty',
    'Tipper,5mmdowndust',
    'Tipper,empty',
    'Trailer,empty'
]

label_dirs = [
    "D:/videos/dataset/labels/train",
    "D:/videos/dataset/labels/val"
]

for label_dir in label_dirs:
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(label_dir, file)
            with open(file_path, "r") as f:
                lines = f.readlines()

            cleaned_lines = []
            for line in lines:
                parts = line.strip().split()

                if len(parts) >= 6:
                    metadata = " ".join(parts[5:])  # Handle any trailing whitespace
                    if metadata in class_names:
                        class_index = class_names.index(metadata)
                        parts[0] = str(class_index)  # Update class id
                        new_line = " ".join(parts[:5]) + "\n"  # Remove metadata
                        cleaned_lines.append(new_line)
                    else:
                        print(f"⚠️ Unknown class label: '{metadata}' in file {file_path}")
                elif len(parts) == 5:
                    # Already cleaned — no metadata
                    cleaned_lines.append(" ".join(parts) + "\n")

            with open(file_path, "w") as f:
                f.writelines(cleaned_lines)

print("✅ Labels updated with correct class indices and metadata removed.")
