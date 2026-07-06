import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_yolo_labels(label_path, img_width, img_height):
    """ Load YOLO labels and convert them to pixel coordinates """
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r") as file:
        for line in file.readlines():
            values = line.strip().split()
            class_id = int(values[0])  # Class ID (pothole = 0)
            x_center, y_center, w, h = map(float, values[1:])

            # Convert normalized coordinates to pixel values
            x1 = int((x_center - w / 2) * img_width)
            y1 = int((y_center - h / 2) * img_height)
            x2 = int((x_center + w / 2) * img_width)
            y2 = int((y_center + h / 2) * img_height)

            labels.append((class_id, x1, y1, x2, y2))
    return labels

def visualize_labels(image_dir, label_dir, output_dir=None):
    """ Load images and corresponding labels, draw bounding boxes, and display/save results """
    
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    image_files = sorted(os.listdir(image_dir))

    for image_name in image_files:
        if not image_name.lower().endswith(('.jpg', '.png')):
            continue  # Skip non-image files

        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
        img_h, img_w = image.shape[:2]

        # Load labels
        labels = load_yolo_labels(label_path, img_w, img_h)

        # Draw bounding boxes
        for _, x1, y1, x2, y2 in labels:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
            cv2.putText(image, "Pothole", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Display image with bounding box
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"Annotated: {image_name}")
        plt.axis("off")
        plt.show()

        # Save visualization if output_dir is provided
        if output_dir:
            output_path = os.path.join(output_dir, image_name)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
            cv2.imwrite(output_path, image_bgr)

    print("Label visualization completed!")

if __name__ == "__main__":
    image_dir = "outputs/wgan/images"
    label_dir = "outputs/wgan/labels"
    output_dir = "outputs/wgan/visualized"  # Set to None if you don't want to save

    visualize_labels(image_dir, label_dir, output_dir)
