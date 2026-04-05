import albumentations as A 
import cv2
import numpy as np
import os 
import matplotlib.pyplot as plt
import warnings 

# Use this script to test how the augementation pipeline works for a single given image

# Mute the Albumentations warning for "Pixel-Only" transforms
warnings.filterwarnings("ignore", message="Got processor for bboxes, but no transform to process it.")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    
    # Test imag efile name 
    base_filename = "0a0c3694-487a156f" 
    
    label_path = os.path.join(root_dir, "data_final/labels/train/", f"{base_filename}.txt")
    image_path = os.path.join(root_dir, "data_final/images/train/", f"{base_filename}.jpg")

    if not os.path.exists(image_path) or not os.path.exists(label_path):
        print(f"Error: Missing files for {base_filename}")
        return

    # Transformation pipeline 
    train_transform = A.Compose([
        # Geometric 
        A.HorizontalFlip(p=0.5), 
        
        # Colour 
        A.RandomBrightnessContrast(p=0.5), 
        A.ColorJitter(p=0.5),                    
        
        # Blur 
        A.MotionBlur(blur_limit=(3, 15), p=0.6),               
        A.GaussianBlur(blur_limit=(0, 3), p=0.6),             
        
    ], bbox_params=A.BboxParams(
        format='yolo', 
        label_fields=['class_labels'],
        # Some bboxes may be slightly out of bounds after augmentation, due to the internal tranfromation to absoulete pixels widths from YOLO format. 
        # They are clipped to be within the image, or removed if they become invalid.
        clip=True, 
        filter_invalid_bboxes=True,  
    ))

    # Load image and convert to RGB for visualization
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load Bounding Boxes 
    if os.path.getsize(label_path) > 0:
        bboxes = np.loadtxt(label_path, delimiter=" ", usecols=range(1, 5)).reshape(-1, 4)
        labels = np.loadtxt(label_path, delimiter=" ", usecols=0, dtype=int).reshape(-1)
    else:
        bboxes, labels = np.empty((0, 4)), np.empty((0,))

    label_map = {0: "traffic_light", 1: "traffic_sign"}
    class_labels = [label_map.get(lbl, "unknown") for lbl in labels]

    # Run Visualization
    visualize_bbox_augmentations(image, bboxes, class_labels, train_transform, samples=3)


# --- Helper Functions ---

def draw_bboxes_yolo(image_np, bboxes, labels, color=(0, 255, 0), thickness=2):
    """Draws YOLO format boxes (normalized [x_center, y_center, w, h]) onto an image."""
    img_res = image_np.copy()
    img_h, img_w = img_res.shape[:2] 

    for bbox, label_name in zip(bboxes, labels):
        x_center, y_center, width, height = bbox
        
        # Convert normalized YOLO to absolute pixel coordinates for OpenCV
        x_min = int((x_center - width / 2) * img_w)
        x_max = int((x_center + width / 2) * img_w)
        y_min = int((y_center - height / 2) * img_h)
        y_max = int((y_center + height / 2) * img_h)

        cv2.rectangle(img_res, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(img_res, str(label_name), (x_min, y_min - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img_res


def visualize_bbox_augmentations(image, bboxes, labels, transform, samples=3):
    """Generates augmented versions and saves a comparison plot to disk."""
    figure, ax = plt.subplots(1, samples + 1, figsize=(15, 5))

    # Original
    ax[0].imshow(draw_bboxes_yolo(image, bboxes, labels))
    ax[0].set_title("Original")
    ax[0].axis("off")

    # Augmented
    for i in range(samples):
        augmented = transform(image=image, bboxes=bboxes, class_labels=labels)
        aug_img = augmented['image']
        aug_box = augmented['bboxes']
        aug_lbl = augmented['class_labels']

        ax[i+1].imshow(draw_bboxes_yolo(aug_img, aug_box, aug_lbl, color=(255, 0, 0)))
        ax[i+1].set_title(f"Augmented {i+1}")
        ax[i+1].axis("off")

    plt.tight_layout()
    
    # Save the result so you can view it in VS Code or via SCP
    output_path = "augmentation_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Success! Plot saved to: {output_path}")

if __name__ == "__main__":
    main()