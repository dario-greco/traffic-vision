import os
import json
from PIL import Image

# 1. Targeted Class Map
# This ensures Traffic Light is always 0 and Traffic Sign is always 1
CLASS_MAP = {
    "traffic light": 0,
    "traffic sign": 1
}

def convert_bdd_to_yolo():
    base_labels_dir = "labels"
    base_images_dir = "images"
    splits = ["train", "val", "test"]
    
    for split in splits:
        split_label_dir = os.path.join(base_labels_dir, split)
        split_image_dir = os.path.join(base_images_dir, split)
        
        if not os.path.exists(split_label_dir):
            continue
            
        print(f"🚀 Processing {split}...")
        processed_count = 0
        
        # Get all JSON files
        json_files = [f for f in os.listdir(split_label_dir) if f.endswith(".json")]
        
        for filename in json_files:
            json_path = os.path.join(split_label_dir, filename)
            txt_path = os.path.join(split_label_dir, filename.replace(".json", ".txt"))
            
            # Map to the corresponding image (assuming .jpg extension)
            image_filename = filename.replace(".json", ".jpg")
            image_path = os.path.join(split_image_dir, image_filename)
            
            # Dynamically get image dimensions to avoid magic numbers
            try:
                with Image.open(image_path) as img:
                    img_w, img_h = img.size
            except FileNotFoundError:
                # Some BDD versions might have different extensions or missing files
                continue
            
            with open(json_path, "r") as f:
                data = json.load(f)
                
            yolo_lines = []
            # BDD Scalabel format: frames[0] contains the labels for the image
            objects = data.get("frames", [{}])[0].get("objects", [])
            
            for obj in objects:
                category = obj.get("category")
                
                # Only process our two target classes
                if category in CLASS_MAP and "box2d" in obj:
                    box = obj["box2d"]
                    
                    # YOLO Normalization Math:
                    # x_center = (x1 + x2) / 2 / width
                    # y_center = (y1 + y2) / 2 / height
                    # w = (x2 - x1) / width
                    # h = (y2 - y1) / height
                    
                    x_center = ((box["x1"] + box["x2"]) / 2.0) / img_w
                    y_center = ((box["y1"] + box["y2"]) / 2.0) / img_h
                    width = (box["x2"] - box["x1"]) / img_w
                    height = (box["y2"] - box["y1"]) / img_h
                    
                    class_id = CLASS_MAP[category]
                    # Append formatted string with 6 decimal precision
                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Save as YOLO .txt file
            with open(txt_path, "w") as f:
                f.writelines(yolo_lines)
                
            processed_count += 1
            
        print(f"✅ Finished {split}: {processed_count} files converted.")

if __name__ == "__main__":
    convert_bdd_to_yolo()