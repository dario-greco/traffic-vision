import os
import shutil
import random

def create_data_final(source_base, dest_base, total_goal=10000):
    # Standard BDD splits
    splits = ['train', 'val', 'test']
    
    # Calculate target counts based on 70/10/20 ratio
    # train = 7000, val = 1000, test = 2000
    targets = {
        'train': int(total_goal * 0.7),
        'val': int(total_goal * 0.1),
        'test': int(total_goal * 0.2)
    }

    print(f"📂 Creating {dest_base} with {total_goal} total images...")

    for split in splits:
        img_src_dir = os.path.join(source_base, 'images', split)
        lbl_src_dir = os.path.join(source_base, 'labels', split)
        
        img_dest_dir = os.path.join(dest_base, 'images', split)
        lbl_dest_dir = os.path.join(dest_base, 'labels', split)

        # Create the new folder structure
        os.makedirs(img_dest_dir, exist_ok=True)
        os.makedirs(lbl_dest_dir, exist_ok=True)

        # 1. Get all available images in this split
        all_images = [f for f in os.listdir(img_src_dir) if f.lower().endswith('.jpg')]
        
        # 2. Shuffle them so the selection is truly random
        random.shuffle(all_images)
        
        # 3. Take only the number we need
        selected_images = all_images[:targets[split]]
        
        print(f"  → Copying {len(selected_images)} images for {split}...")

        for img_name in selected_images:
            # Copy the Image
            shutil.copy2(
                os.path.join(img_src_dir, img_name), 
                os.path.join(img_dest_dir, img_name)
            )
            
            # Copy the corresponding Label (.txt)
            label_name = img_name.rsplit('.', 1)[0] + '.txt'
            src_label_path = os.path.join(lbl_src_dir, label_name)
            
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, os.path.join(lbl_dest_dir, label_name))
            else:
                # Create an empty file if no labels exist (Negative Sample)
                open(os.path.join(lbl_dest_dir, label_name), 'a').close()

    print(f"\n✅ Done! Your 10k subset is ready in: {os.path.abspath(dest_base)}")

if __name__ == "__main__":
    # Assumes your current folder has the 'images' and 'labels' folders inside it
    create_data_final('.', 'data_final', 10000)