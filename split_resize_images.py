import os
import cv2
import numpy as np

def split_and_resize(image, mask, base_filename, save_image_dir, save_mask_dir, new_dim=512):
    h, w = image.shape[:2]
    split_size = h // 3  # Splitting into 3x3 grid

    for i in range(3):
        for j in range(3):
            sub_image = image[i * split_size:(i + 1) * split_size, j * split_size:(j + 1) * split_size]
            sub_mask = mask[i * split_size:(i + 1) * split_size, j * split_size:(j + 1) * split_size]

            resized_image = cv2.resize(sub_image, (new_dim, new_dim))
            resized_mask = cv2.resize(sub_mask, (new_dim, new_dim))

            sub_filename = f"{base_filename}_{i}_{j}.png"
            cv2.imwrite(os.path.join(save_image_dir, sub_filename), resized_image)
            cv2.imwrite(os.path.join(save_mask_dir, sub_filename), resized_mask)

def process_directory(image_dir, mask_dir, save_image_dir, save_mask_dir, new_dim=512):
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is not None and mask is not None:
                base_filename = os.path.splitext(filename)[0]
                split_and_resize(image, mask, base_filename, save_image_dir, save_mask_dir, new_dim)

# Define paths
base_dir = 'datasets/data/png'
process_directory(os.path.join(base_dir, 'train'), os.path.join(base_dir, 'train_labels'), os.path.join(base_dir, 'train_split_and_txt'), os.path.join(base_dir, 'train_split_labels'))
process_directory(os.path.join(base_dir, 'val'), os.path.join(base_dir, 'val_labels'), os.path.join(base_dir, 'val_split_and_txt'), os.path.join(base_dir, 'val_split_labels'))
process_directory(os.path.join(base_dir, 'test'), os.path.join(base_dir, 'test_labels'), os.path.join(base_dir, 'test_split_and_txt'), os.path.join(base_dir, 'test_split_labels'))
