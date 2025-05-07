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
        os.makedirs(os.path.join(base_path, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'test', class_name), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'val', class_name), exist_ok=True)

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

def augment_image(image, augmentation_count=5):
    """Apply random augmentations to create multiple variations of the image."""
    augmented_images = []
    
    for _ in range(augmentation_count):
        aug_img = image.copy()
        
        # Apply random augmentations with some probability
        if random.random() > 0.3:  # 70% chance to apply crop
            aug_img = random_crop(aug_img)
            
        if random.random() > 0.3:  # 70% chance to apply rotation
            aug_img = random_rotation(aug_img)
            
        if random.random() > 0.3:  # 70% chance to apply brightness adjustment
            aug_img = random_brightness(aug_img)
            
        if random.random() > 0.3:  # 70% chance to apply noise
            aug_img = random_noise(aug_img)
            
        augmented_images.append(aug_img)
        
    return augmented_images

def process_classification_data(src_root, dst_root, augmentation_count=5):
    """Process and augment classification dataset."""
    # Collect all image paths
    class_dirs = [d for d in os.listdir(os.path.join(src_root, 'Classification')) 
                  if os.path.isdir(os.path.join(src_root, 'Classification', d))]
    
    all_images = []
    class_counts = {}
    
    for class_dir in class_dirs:
        class_path = os.path.join(src_root, 'Classification', class_dir)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        for img_file in image_files:
            all_images.append((os.path.join(class_path, img_file), class_dir))
        
        class_counts[class_dir] = len(image_files)
    
    # Split into train, test, validation
    train_imgs, test_val_imgs = train_test_split(all_images, test_size=0.4, stratify=[img[1] for img in all_images], random_state=42)
    test_imgs, val_imgs = train_test_split(test_val_imgs, test_size=0.5, stratify=[img[1] for img in test_val_imgs], random_state=42)
    
    # Create augmented images
    aug_counts = {}
    for class_dir in class_dirs:
        aug_counts[class_dir] = 0
    
    # Process train set with augmentation
    for img_path, class_name in tqdm(train_imgs, desc="Processing training images"):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Save original image
        filename = os.path.basename(img_path)
        dst_path = os.path.join(dst_root, 'train', class_name, filename)
        cv2.imwrite(dst_path, img)
        
        # Generate and save augmented images
        augmented = augment_image(img, augmentation_count)
        for i, aug_img in enumerate(augmented):
            aug_filename = f"{os.path.splitext(filename)[0]}_aug{i+1}{os.path.splitext(filename)[1]}"
            aug_path = os.path.join(dst_root, 'train', class_name, aug_filename)
            cv2.imwrite(aug_path, aug_img)
            aug_counts[class_name] += 1
    
    # Process test set (no augmentation)
    for img_path, class_name in tqdm(test_imgs, desc="Processing test images"):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Save original image only
        filename = os.path.basename(img_path)
        dst_path = os.path.join(dst_root, 'test', class_name, filename)
        cv2.imwrite(dst_path, img)
    
    # Process validation set (no augmentation)
    for img_path, class_name in tqdm(val_imgs, desc="Processing validation images"):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Save original image only
        filename = os.path.basename(img_path)
        dst_path = os.path.join(dst_root, 'val', class_name, filename)
        cv2.imwrite(dst_path, img)
    
    return class_counts, aug_counts

def main():
    # Configuration
    src_root = "."  # Current directory
    dst_root = "./augmented_dataset/Classification"
    augmentation_count = 5  # Number of augmented images to generate per original image
    
    # Create destination directories
    create_directories(dst_root)
    
    # Process data
    print("=== Processing Classification Dataset ===")
    class_counts, class_aug_counts = process_classification_data(src_root, dst_root, augmentation_count)
    
    # Print statistics
    print("\n=== Classification Dataset Statistics ===")
    print(f"{'Class':<20} {'Original':<10} {'Augmented':<10} {'Total':<10}")
    print("-" * 50)
    
    total_orig_class = 0
    total_aug_class = 0
    
    for class_name in class_counts:
        orig = class_counts[class_name]
        aug = class_aug_counts.get(class_name, 0)
        total = orig + aug
        print(f"{class_name:<20} {orig:<10} {aug:<10} {total:<10}")
        total_orig_class += orig
        total_aug_class += aug
    
    print("-" * 50)
    print(f"{'TOTAL':<20} {total_orig_class:<10} {total_aug_class:<10} {total_orig_class + total_aug_class:<10}")
    
    print("\nTotal Dataset Size Increase:")
    print(f"Original: {total_orig_class}")
    print(f"Augmented: {total_aug_class}")
    print(f"Total: {total_orig_class + total_aug_class}")
    print(f"Multiplier: {(total_orig_class + total_aug_class) / total_orig_class:.2f}x")

if __name__ == "__main__":
    main()