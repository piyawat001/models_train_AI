import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

def create_directories(base_path):
    """Create necessary directories for the augmented dataset."""
    for class_name in ['Ameloblastoma', 'Dentigerous cyst', 'Normal jaw', 'OKC']:
        for split in ['train', 'test', 'val']:
            os.makedirs(os.path.join(base_path, split, class_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(base_path, split, class_name, 'labels'), exist_ok=True)

def random_crop(image, min_crop_ratio=0.8):
    """Randomly crop image with crop ratio between min_crop_ratio and 1.0."""
    h, w = image.shape[:2]
    
    # Calculate crop dimensions
    crop_ratio = random.uniform(min_crop_ratio, 1.0)
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    
    # Calculate random crop position
    top = random.randint(0, h - crop_h) if h > crop_h else 0
    left = random.randint(0, w - crop_w) if w > crop_w else 0
    
    # Crop the image
    cropped = image[top:top+crop_h, left:left+crop_w]
    
    # Resize back to original size if cropped
    if crop_ratio < 1.0:
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
    return cropped

def random_rotation(image, max_angle=15):
    """Randomly rotate image between -max_angle and +max_angle degrees."""
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    
    # Calculate rotation matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return rotated

def random_brightness(image, max_delta=50):
    """Randomly adjust brightness between -max_delta and +max_delta."""
    delta = random.uniform(-max_delta, max_delta)
    adjusted = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    return adjusted

def random_noise(image, max_noise_percent=0.2):
    """Add random noise to the image up to max_noise_percent."""
    h, w = image.shape[:2]
    noise_level = random.uniform(0, max_noise_percent)
    
    # Create noise image
    noise = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    noise = cv2.GaussianBlur(noise, (7, 7), 0)
    
    # Scale noise by the desired level
    noise = (noise * noise_level).astype(np.uint8)
    
    # Add noise to the image
    if len(image.shape) == 3:  # Color image
        noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    
    noisy = cv2.addWeighted(image, 1, noise, 0.5, 0)
    return noisy

def process_segmentation_data(src_root, dst_root, augmentation_count=5):
    """Process and augment segmentation dataset."""
    # Collect all image-label pairs
    class_dirs = [d for d in os.listdir(os.path.join(src_root, 'Segmentation')) 
                  if os.path.isdir(os.path.join(src_root, 'Segmentation', d))]
    
    print(f"Found class directories: {class_dirs}")
    
    all_pairs = []
    class_counts = {}
    
    for class_dir in class_dirs:
        class_path = os.path.join(src_root, 'Segmentation', class_dir)
        img_dir = os.path.join(class_path, 'images')
        label_dir = os.path.join(class_path, 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            print(f"Warning: Missing images or labels directory for {class_dir}")
            continue
        
        # List all files in image and label directories for debugging
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        label_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        print(f"Class: {class_dir}")
        print(f"  - Images directory: {img_dir}")
        print(f"  - Found {len(image_files)} image files")
        print(f"  - Labels directory: {label_dir}")
        print(f"  - Found {len(label_files)} label files")
        
        if not image_files:
            print(f"  - Warning: No image files found in {img_dir}")
        if not label_files:
            print(f"  - Warning: No label files found in {label_dir}")
        
        # Display some sample filenames for debugging
        if image_files:
            print(f"  - Sample image filenames: {image_files[:3]}")
        if label_files:
            print(f"  - Sample label filenames: {label_files[:3]}")
        
        valid_pairs = []
        
        # Check if labels might have different extensions
        all_label_files = set(os.listdir(label_dir))
        
        for img_file in image_files:
            # First try exact match
            if img_file in all_label_files:
                valid_pairs.append((
                    os.path.join(img_dir, img_file),
                    os.path.join(label_dir, img_file),
                    class_dir
                ))
                continue
            
            # Try matching base name with different extension
            base_name = os.path.splitext(img_file)[0]
            matching_labels = [f for f in all_label_files if f.startswith(base_name + '.')]
            
            if matching_labels:
                valid_pairs.append((
                    os.path.join(img_dir, img_file),
                    os.path.join(label_dir, matching_labels[0]),
                    class_dir
                ))
        
        print(f"  - Found {len(valid_pairs)} valid image-label pairs")
        
        all_pairs.extend(valid_pairs)
        class_counts[class_dir] = len(valid_pairs)
    
    # Check if we found any valid pairs
    if len(all_pairs) == 0:
        print("No valid image-label pairs found in the segmentation dataset.")
        print("Please check that:")
        print("1. The Segmentation directory contains class subdirectories")
        print("2. Each class subdirectory has 'images' and 'labels' folders")
        print("3. Image and label files have matching filenames (or base names)")
        print("4. Files have supported extensions (.png, .jpg, .jpeg, .tif, .tiff)")
        return {}, {}
    
    # Split into train, test, validation
    train_pairs, test_val_pairs = train_test_split(all_pairs, test_size=0.4, stratify=[p[2] for p in all_pairs], random_state=42)
    test_pairs, val_pairs = train_test_split(test_val_pairs, test_size=0.5, stratify=[p[2] for p in test_val_pairs], random_state=42)
    
    # Create augmented pairs
    aug_counts = {}
    for class_dir in class_dirs:
        aug_counts[class_dir] = 0
    
    # Process train set with augmentation
    for img_path, label_path, class_name in tqdm(train_pairs, desc="Processing training segmentation pairs"):
        img = cv2.imread(img_path)
        label = cv2.imread(label_path)
        
        if img is None:
            print(f"Warning: Could not read image: {img_path}")
            continue
            
        if label is None:
            print(f"Warning: Could not read label: {label_path}")
            continue
            
        # Save original pair
        img_filename = os.path.basename(img_path)
        label_filename = os.path.basename(label_path)
        
        dst_img_path = os.path.join(dst_root, 'train', class_name, 'images', img_filename)
        dst_label_path = os.path.join(dst_root, 'train', class_name, 'labels', label_filename)
        
        cv2.imwrite(dst_img_path, img)
        cv2.imwrite(dst_label_path, label)
        
        # Generate and save augmented pairs
        # Note: We need to apply the exact same augmentation to both image and label
        for i in range(augmentation_count):
            # Create a random seed for this augmentation set
            seed = random.randint(0, 10000)
            
            # Temporarily set the seed for reproducible augmentation between image and label
            random.seed(seed)
            np.random.seed(seed)
            aug_img = img.copy()
            
            random.seed(seed)
            np.random.seed(seed)
            aug_label = label.copy()
            
            # Apply the same random transformations
            if random.random() > 0.3:
                aug_img = random_crop(aug_img)
                aug_label = random_crop(aug_label)
                
            if random.random() > 0.3:
                aug_img = random_rotation(aug_img)
                aug_label = random_rotation(aug_label)
            
            # Only apply these to the image, not the label
            if random.random() > 0.3:
                aug_img = random_brightness(aug_img)
                
            if random.random() > 0.3:
                aug_img = random_noise(aug_img)
            
            # Save augmented pair
            aug_img_filename = f"{os.path.splitext(img_filename)[0]}_aug{i+1}{os.path.splitext(img_filename)[1]}"
            aug_label_filename = f"{os.path.splitext(label_filename)[0]}_aug{i+1}{os.path.splitext(label_filename)[1]}"
            
            aug_img_path = os.path.join(dst_root, 'train', class_name, 'images', aug_img_filename)
            aug_label_path = os.path.join(dst_root, 'train', class_name, 'labels', aug_label_filename)
            
            cv2.imwrite(aug_img_path, aug_img)
            cv2.imwrite(aug_label_path, aug_label)
            
            aug_counts[class_name] += 1
            
            # Reset the random seed
            random.seed()
            np.random.seed()
    
    # Process test set (no augmentation)
    for img_path, label_path, class_name in tqdm(test_pairs, desc="Processing test segmentation pairs"):
        img = cv2.imread(img_path)
        label = cv2.imread(label_path)
        
        if img is None or label is None:
            continue
            
        # Save original pair only
        img_filename = os.path.basename(img_path)
        label_filename = os.path.basename(label_path)
        
        dst_img_path = os.path.join(dst_root, 'test', class_name, 'images', img_filename)
        dst_label_path = os.path.join(dst_root, 'test', class_name, 'labels', label_filename)
        
        cv2.imwrite(dst_img_path, img)
        cv2.imwrite(dst_label_path, label)
    
    # Process validation set (no augmentation)
    for img_path, label_path, class_name in tqdm(val_pairs, desc="Processing validation segmentation pairs"):
        img = cv2.imread(img_path)
        label = cv2.imread(label_path)
        
        if img is None or label is None:
            continue
            
        # Save original pair only
        img_filename = os.path.basename(img_path)
        label_filename = os.path.basename(label_path)
        
        dst_img_path = os.path.join(dst_root, 'val', class_name, 'images', img_filename)
        dst_label_path = os.path.join(dst_root, 'val', class_name, 'labels', label_filename)
        
        cv2.imwrite(dst_img_path, img)
        cv2.imwrite(dst_label_path, label)
    
    return class_counts, aug_counts

def main():
    # Configuration
    src_root = "."  # Current directory
    dst_root = "./augmented_dataset/Segmentation"
    augmentation_count = 5  # Number of augmented images to generate per original image
    
    # Create destination directories
    create_directories(dst_root)
    
    # Process data
    print("=== Processing Segmentation Dataset ===")
    seg_counts, seg_aug_counts = process_segmentation_data(src_root, dst_root, augmentation_count)
    
    # Check if we found any data
    if not seg_counts:
        print("No segmentation data was processed. Please make sure the Segmentation directory exists with the required structure.")
        return
    
    # Print statistics
    print("\n=== Segmentation Dataset Statistics ===")
    print(f"{'Class':<20} {'Original':<10} {'Augmented':<10} {'Total':<10}")
    print("-" * 50)
    
    total_orig_seg = 0
    total_aug_seg = 0
    
    for class_name in seg_counts:
        orig = seg_counts[class_name]
        aug = seg_aug_counts.get(class_name, 0)
        total = orig + aug
        print(f"{class_name:<20} {orig:<10} {aug:<10} {total:<10}")
        total_orig_seg += orig
        total_aug_seg += aug
    
    print("-" * 50)
    print(f"{'TOTAL':<20} {total_orig_seg:<10} {total_aug_seg:<10} {total_orig_seg + total_aug_seg:<10}")
    
    print("\nTotal Dataset Size Increase:")
    print(f"Original: {total_orig_seg}")
    print(f"Augmented: {total_aug_seg}")
    print(f"Total: {total_orig_seg + total_aug_seg}")
    print(f"Multiplier: {(total_orig_seg + total_aug_seg) / total_orig_seg:.2f}x" if total_orig_seg > 0 else "Multiplier: N/A (no original data)")

if __name__ == "__main__":
    main()