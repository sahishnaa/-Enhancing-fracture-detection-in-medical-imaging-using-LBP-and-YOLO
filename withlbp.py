from google.colab import files
uploaded = files.upload()
import os

label_dir = "/content/dataset/Break bone.v5i.yolov8-obb/train/labels"
extra_labels = [
    "Spiral-2-_jpg.rf.31fe43ae7d19babc8e83cb5b1fff8241",
    "Spiral-22-_jpg.rf.33f03248123b922b37a72b0296e585b0",
    "Spiral-23-_jpg.rf.0a8e35a0ba8cc000fe5a185183087810",
    "Spiral-3-_jpg.rf.43f89fd52ee909cf043bca5f0d2fd850",
    "Spiral-5-_jpg.rf.2c95f448e3bd7879c81a9215b31e9ba7",
    "Spiral_1_jpg.rf.88fa66b37c2c2f701ef58450dd46ee14",
    "Spiral_jpg.rf.fa0caabe2e46e5cc7a74432fd946d6f1"
]

for name in extra_labels:
    path = os.path.join(label_dir, name + ".txt")
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted: {path}")
    else:
        print(f"Not found: {path}")
import os

# Define paths using raw strings to avoid issues with spaces
image_dir = r"dataset/Break bone.v5i.yolov8-obb/train/images"
label_dir = r"dataset/Break bone.v5i.yolov8-obb/train/labels"

# Count files
num_images = len(os.listdir(image_dir))
num_labels = len(os.listdir(label_dir))

print(f"Images: {num_images}")
print(f"Labels: {num_labels}")
!pip install ultralytics
from ultralytics import YOLO
!pip install ultralytics opencv-python scikit-learn matplotlib

import os
import cv2
import numpy as np
import shutil
from ultralytics import YOLO
from collections import Counter
from skimage.feature import local_binary_pattern

# Step 3: Define original and new dataset paths
original_base = '/content/dataset/Break bone.v5i.yolov8-obb'
lbp_base = '/content/dataset_lbp/Break bone.v5i.yolov8-obb'

sets = ['train', 'valid', 'test']
image_dirs = {s: os.path.join(original_base, s, 'images') for s in sets}
label_dirs = {s: os.path.join(original_base, s, 'labels') for s in sets}

lbp_image_dirs = {s: os.path.join(lbp_base, s, 'images') for s in sets}
lbp_label_dirs = {s: os.path.join(lbp_base, s, 'labels') for s in sets}

# Create directories for LBP dataset
for s in sets:
    os.makedirs(lbp_image_dirs[s], exist_ok=True)
    os.makedirs(lbp_label_dirs[s], exist_ok=True)

# Step 4: Define LBP function
def apply_lbp(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error reading image: {image_path}")
        return None
    lbp_image = local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_image = ((lbp_image / lbp_image.max()) * 255).astype(np.uint8)
    return lbp_image

# Step 5: Apply LBP and prepare dataset
class_counts = Counter()
total_images = 0

for split in sets:
    image_dir = image_dirs[split]
    label_dir = label_dirs[split]
    out_img_dir = lbp_image_dirs[split]
    out_lbl_dir = lbp_label_dirs[split]

    for file in os.listdir(image_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, file)
            label_path = os.path.join(label_dir, file.rsplit('.', 1)[0] + '.txt')

            if os.path.exists(label_path):
                lbp_img = apply_lbp(img_path)
                if lbp_img is not None:
                    cv2.imwrite(os.path.join(out_img_dir, file), lbp_img)
                    shutil.copy(label_path, os.path.join(out_lbl_dir, os.path.basename(label_path)))
                    total_images += 1
                    class_counts[file.rsplit('_', 1)[0]] += 1
            else:
                print(f"Label missing for: {img_path}")

print(f"\n✅ Total images processed: {total_images}")

# Step 6: Copy and update YAML
original_yaml_path = os.path.join(original_base, 'data.yaml')
new_yaml_path = os.path.join(lbp_base, 'data.yaml')

with open(original_yaml_path, 'r') as f:
    yaml_lines = f.readlines()

# Replace path line
with open(new_yaml_path, 'w') as f:
    for line in yaml_lines:
        if line.startswith('path:'):
            f.write(f"path: {lbp_base}\n")
        else:
            f.write(line)

print(f"✅ YAML updated at: {new_yaml_path}")

# Step 7: Train YOLO on LBP dataset
model = YOLO('yolov8n.pt')
results = model.train(data=new_yaml_path, epochs=100, imgsz=640)

# Step 8: Evaluate
val_results = model.val()
print("📈 Validation results:", val_results)
