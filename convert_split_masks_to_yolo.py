import os
import cv2
import numpy as np

# Define paths
base_dir = 'datasets/data/png'
train_mask_path = os.path.join(base_dir, 'train_split_labels')
train_label_path = os.path.join(base_dir, 'train_split_and_txt')  # Use train_split_and_txt directory for labels

val_mask_path = os.path.join(base_dir, 'val_split_labels')
val_label_path = os.path.join(base_dir, 'val_split_and_txt')  # Use val_split_and_txt directory for labels

test_mask_path = os.path.join(base_dir, 'test_split_labels')
test_label_path = os.path.join(base_dir, 'test_split_and_txt')  # Use test_split_and_txt directory for labels

# New dimensions
new_dim = 512

def create_yolo_label(mask, original_shape, resized_shape, class_id=0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = original_shape[:2]
    new_h, new_w = resized_shape[:2]
    yolo_labels = []
    
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        
        # Adjust coordinates to the resized image
        x = x * new_w / w
        y = y * new_h / h
        width = width * new_w / w
        height = height * new_h / h
        
        x_center = (x + width / 2) / new_w
        y_center = (y + height / 2) / new_h
        norm_width = width / new_w
        norm_height = height / new_h
        
        yolo_labels.append(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}")
    
    return yolo_labels

def convert_masks_to_yolo_labels(mask_dir, label_dir, new_dim):
    os.makedirs(label_dir, exist_ok=True)
    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith('.png'):
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                # No need to resize image and mask again
                yolo_labels = create_yolo_label(mask, (new_dim, new_dim), (new_dim, new_dim))
                label_file = os.path.join(label_dir, mask_file.replace('.png', '.txt'))
                
                with open(label_file, 'w') as f:
                    f.write("\n".join(yolo_labels))

# Convert masks to YOLO labels
convert_masks_to_yolo_labels(train_mask_path, train_label_path, new_dim)
convert_masks_to_yolo_labels(val_mask_path, val_label_path, new_dim)
convert_masks_to_yolo_labels(test_mask_path, test_label_path, new_dim)
