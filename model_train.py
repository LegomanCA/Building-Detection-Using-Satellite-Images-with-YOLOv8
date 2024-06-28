import logging
import os
from ultralytics import YOLO
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_without_labels_and_confidences(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for result in results:
        img = result.orig_img  # Original image
        boxes = result.boxes  # Detected bounding boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

        img_filename = os.path.join(save_dir, os.path.basename(result.path))
        cv2.imwrite(img_filename, img)

def main():
    logger.info("Starting YOLOv8 training script...")

    # Load a pre-trained YOLOv8 model
    logger.info("Loading pre-trained YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # YOLOv8 nano model

    # Train the model on your dataset
    logger.info("Starting training...")
    model.train(
        data='dataset.yaml',
        epochs=50,
        imgsz=512,
        batch=16,  # Increased batch size due to increased number of images
        device='0',
        amp=True,
        save_period=-1,
        workers=4,
        project='runs/detect',
        name='train_experiment3'
    )

    # Evaluate the model
    logger.info("Evaluating the model...")
    model.val(data='dataset.yaml', batch=16, device='0')

    # Make predictions on new images
    logger.info("Making predictions...")
    results = model.predict(source='datasets/data/png/test_split_and_txt', save=False, show=False)

    # Save results with custom visualization options
    plot_without_labels_and_confidences(results, 'runs/detect/train_experiment3/predictions')

    logger.info("Training script completed successfully.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # This is only needed for Windows
    main()
