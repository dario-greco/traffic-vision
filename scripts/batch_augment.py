import albumentations as A 
import cv2
import numpy as np
import os 
import random
import warnings 

# Mute the "Pixel-Only" transform warning for a clean terminal output
warnings.filterwarnings("ignore", message="Got processor for bboxes, but no transform to process it.")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def batch_augment_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    
    # 1. Define Input and Output Directories
    in_img_dir = os.path.join(root_dir, "data_final/images/train")
    in_lbl_dir = os.path.join(root_dir, "data_final/labels/train")
    
    out_img_dir = os.path.join(root_dir, "data_augmented/images/train")
    out_lbl_dir = os.path.join(root_dir, "data_augmented/labels/train")
    
    # Create the new output directories if they don't exist
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # Transform Pipeline
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5), 
        A.RandomBrightnessContrast(p=0.5), 
        A.ColorJitter(p=0.5),                    
        A.MotionBlur(blur_limit=(3, 13), p=0.5),               
        A.GaussianBlur(blur_limit=(0, 3), p=0.5),             
    ], bbox_params=A.BboxParams(
        format='yolo', 
        label_fields=['class_labels'],
        clip=True,
        filter_invalid_bboxes=True,
    ))

    image_files = [f for f in os.listdir(in_img_dir) if f.endswith('.jpg')]
    total_files = len(image_files)
    print(f"🚀 Found {total_files} images. Starting offline batch augmentation...")

    # 3. Process Every Image
    for i, filename in enumerate(image_files):
        base_name = os.path.splitext(filename)[0]
        
        img_path = os.path.join(in_img_dir, filename)
        lbl_path = os.path.join(in_lbl_dir, f"{base_name}.txt")
        
        # Append "_aug" to the output filenames to avoid confusion
        out_img_path = os.path.join(out_img_dir, f"{base_name}_aug.jpg")
        out_lbl_path = os.path.join(out_lbl_dir, f"{base_name}_aug.txt")

        # Load Image (Convert OpenCV BGR to standard RGB)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load Labels
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            bboxes = np.loadtxt(lbl_path, delimiter=" ", usecols=range(1, 5)).reshape(-1, 4)
            labels = np.loadtxt(lbl_path, delimiter=" ", usecols=0, dtype=int).reshape(-1)
        else:
            # Handle images with no traffic lights or signs gracefully
            bboxes, labels = np.empty((0, 4)), np.empty((0,), dtype=int)

        # Apply Augmentation (clip=True handles the edge cases automatically)
        augmented = train_transform(image=image, bboxes=bboxes, class_labels=labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        # 4. Save the Augmented Data
        # Convert back to BGR so OpenCV saves the colors correctly
        cv2.imwrite(out_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        
        with open(out_lbl_path, 'w') as f:
            for bbox, cls in zip(aug_bboxes, aug_labels):
                f.write(f"{int(cls)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        # Print progress to the terminal
        if (i + 1) % 500 == 0:
            print(f"✅ Processed {i + 1}/{total_files} images...")

    print("🎉 Batch augmentation complete! All data saved to data_augmented/")

if __name__ == "__main__":
    batch_augment_dataset()