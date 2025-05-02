from google.colab import files
uploaded = files.upload()
!unzip newdataset.zip -d dataset
import os

# Define paths using raw strings to avoid issues with spaces
image_dir = r"dataset/Break bone.v5i.yolov8-obb/train/images"
label_dir = r"dataset/Break bone.v5i.yolov8-obb/train/labels"

# Count files
num_images = len(os.listdir(image_dir))
num_labels = len(os.listdir(label_dir))

print(f"Images: {num_images}")
print(f"Labels: {num_labels}")
import os

image_dir = "/content/dataset/Break bone.v5i.yolov8-obb/train/images"
label_dir = "/content/dataset/Break bone.v5i.yolov8-obb/train/labels"

image_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir)]
label_files = [os.path.splitext(f)[0] for f in os.listdir(label_dir)]

unmatched_images = set(image_files) - set(label_files)

print(f"Unmatched image(s): {unmatched_images}")
image_path = "/content/dataset/Break bone.v5i.yolov8-obb/train/images/247_2019_4591_Fig4_HTML_jpg.rf.471a719b4e635c6bac6e548948156884.jpg"

import os
if os.path.exists(image_path):
    os.remove(image_path)
    print(f"Deleted: {image_path}")
image_dir = "/content/dataset/Break bone.v5i.yolov8-obb/train/images"
label_dir = "/content/dataset/Break bone.v5i.yolov8-obb/train/labels"

image_files = set(os.path.splitext(f)[0] for f in os.listdir(image_dir))
label_files = set(os.path.splitext(f)[0] for f in os.listdir(label_dir))

# Labels without matching images
extra_labels = label_files - image_files

print(f"Extra label files: {len(extra_labels)}")
for label in sorted(extra_labels):
    print(label + ".txt")
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
!pip install ultralytics
from ultralytics import YOLO
# Define dataset path
dataset_path = "dataset" # Adjust if needed

# Load YOLOv8 model (smallest version first)
model = YOLO("yolov8n.yaml") # You can also try yolov8s.yaml or yolov8m.yaml

# Train the model
model.train(data="/content/dataset/Break bone.v5i.yolov8-obb/data.yaml", epochs=100, imgsz=640)
