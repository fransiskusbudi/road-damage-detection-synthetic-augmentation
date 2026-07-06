# import os
# import cv2
# import numpy as np
# import random
# from tqdm import tqdm

# def get_largest_contour(mask):
#     """ Find the largest contour in the mask and return it """
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None  # No valid region found

#     # Sort contours by area and pick the largest one
#     largest_contour = max(contours, key=cv2.contourArea)
#     return largest_contour

# def is_pothole_inside_contour(contour, x1, y1, x2, y2):
#     """ Check if the whole pothole fits inside the largest contour """
#     points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]  # 4 corners
#     return all(cv2.pointPolygonTest(contour, pt, False) >= 0 for pt in points)

# def get_valid_pothole_location(contour, mask_shape, pothole_size, filename):
#     """ Find a valid location where the full pothole fits inside the contour """
#     attempts = 100  # Retry limit
#     for _ in range(attempts):
#         x = random.randint(0, mask_shape[1] - pothole_size)
#         y = random.randint(0, mask_shape[0] - pothole_size)
#         x1, y1, x2, y2 = x, y, x + pothole_size, y + pothole_size

#         if is_pothole_inside_contour(contour, x1, y1, x2, y2):
#             return x1, y1, x2, y2  # Valid placement found

#     print(f"Warning: Could not find a valid placement inside the contour for {filename}")
#     return None  # If no valid location is found

# def poisson_blend_pothole(image, mask, pothole, filename):
#     """ Insert a pothole strictly inside the largest drivable area using Poisson blending """

#     # Convert mask to binary
#     _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

#     # Get the largest contour
#     largest_contour = get_largest_contour(binary_mask)
#     if largest_contour is None:
#         print(f"No valid drivable area found in the mask for {filename}.")
#         return image, None, None, None

#     # Randomly choose a size between 32x32 and 64x64
#     pothole_size = random.randint(32, 64)
#     pothole_resized = cv2.resize(pothole, (pothole_size, pothole_size))

#     # Find a fully valid location inside the contour
#     location = get_valid_pothole_location(largest_contour, mask.shape, pothole_size, filename)
#     if location is None:
#         return image, None, None, None

#     x1, y1, x2, y2 = location

#     # Ensure pothole fits inside image bounds
#     pothole_resized = pothole_resized[:y2 - y1, :x2 - x1]

#     # Create pothole mask (white where pothole exists)
#     pothole_gray = cv2.cvtColor(pothole_resized, cv2.COLOR_RGB2GRAY)
#     _, pothole_mask = cv2.threshold(pothole_gray, 10, 255, cv2.THRESH_BINARY)

#     # Define center for Poisson blending
#     center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

#     # Perform seamless cloning (Poisson blending)
#     blended_image = cv2.seamlessClone(pothole_resized, image, pothole_mask, center, cv2.NORMAL_CLONE)

#     # Convert bounding box coordinates to YOLO format (normalized)
#     img_h, img_w = image.shape[:2]
#     x_center = (x1 + x2) / 2 / img_w
#     y_center = (y1 + y2) / 2 / img_h
#     w_norm = (x2 - x1) / img_w
#     h_norm = (y2 - y1) / img_h

#     # YOLO annotation format: "class_id x_center y_center width height"
#     label_data = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"

#     return blended_image, label_data, largest_contour, pothole_size

# def process_dataset(image_dir, mask_dir, pothole_dir, output_dir):
#     """ Process all images, blend potholes using masks, save images, and generate labels """

#     image_output_dir = os.path.join(output_dir, "images")
#     label_output_dir = os.path.join(output_dir, "labels")
#     visualized_output_dir = os.path.join(output_dir, "visualized")

#     os.makedirs(image_output_dir, exist_ok=True)
#     os.makedirs(label_output_dir, exist_ok=True)
#     os.makedirs(visualized_output_dir, exist_ok=True)

#     image_files = sorted(os.listdir(image_dir))
#     pothole_files = sorted(os.listdir(pothole_dir))

#     for image_name in tqdm(image_files, desc="Processing images"):
#         if not image_name.lower().endswith(('.jpg', '.png')):
#             continue

#         image_path = os.path.join(image_dir, image_name)
#         mask_path = os.path.join(mask_dir, 'da_' + image_name)

#         if not os.path.exists(mask_path):
#             print(f"Mask not found for {image_name}, skipping.")
#             continue

#         # Read images
#         image = cv2.imread(image_path)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#         # Choose a random pothole
#         pothole_name = random.choice(pothole_files)
#         pothole_path = os.path.join(pothole_dir, pothole_name)
#         pothole = cv2.imread(pothole_path, cv2.IMREAD_UNCHANGED)  # Keep alpha channel if exists

#         # Convert pothole to RGB if it has an alpha channel
#         if pothole.shape[-1] == 4:
#             pothole = cv2.cvtColor(pothole, cv2.COLOR_BGRA2BGR)

#         # Blend pothole into the image and get annotation data
#         blended_image, label_data, largest_contour, pothole_size = poisson_blend_pothole(image, mask, pothole, image_name)

#         if label_data is not None:
#             # Save the blended image
#             output_image_path = os.path.join(image_output_dir, image_name)
#             cv2.imwrite(output_image_path, blended_image)

#             # Save the label file
#             label_filename = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
#             output_label_path = os.path.join(label_output_dir, label_filename)
#             with open(output_label_path, "w") as label_file:
#                 label_file.write(label_data)

#             # Visualize labels and contours
#             visualize_labels(blended_image, label_data, largest_contour, pothole_size, os.path.join(visualized_output_dir, image_name))

#     print(f"Processed images saved in {image_output_dir}")
#     print(f"Annotations saved in {label_output_dir}")
#     print(f"Visualized labels saved in {visualized_output_dir}")

# def visualize_labels(image, label_data, largest_contour, pothole_size, output_path):
#     """ Draw bounding boxes, pothole size, and largest contour on the image """

#     for line in label_data.strip().split("\n"):
#         values = line.split()
#         x_center, y_center, w, h = map(float, values[1:])

#         # Convert to pixel coordinates
#         x1, y1 = int((x_center - w / 2) * image.shape[1]), int((y_center - h / 2) * image.shape[0])
#         x2, y2 = int((x_center + w / 2) * image.shape[1]), int((y_center + h / 2) * image.shape[0])

#         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         cv2.putText(image, f"Pothole: {pothole_size}x{pothole_size}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#     if largest_contour is not None:
#         cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)  # Draw contour in green

#     cv2.imwrite(output_path, image)

# if __name__ == "__main__":
#     process_dataset("bdd_selected_250/images", "bdd_selected_250/mask", "diffusion_model_generated_1000", "outputs/bdd_diffusion_250")

# import os
# import cv2
# import numpy as np
# import random
# from tqdm import tqdm

# def get_largest_contour(mask):
#     """ Find the largest contour in the mask and return it """
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None  # No valid region found

#     # Sort contours by area and pick the largest one
#     largest_contour = max(contours, key=cv2.contourArea)
#     return largest_contour

# def is_pothole_inside_contour(contour, x1, y1, x2, y2):
#     """ Check if the whole pothole fits inside the largest contour """
#     points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]  # 4 corners
#     return all(cv2.pointPolygonTest(contour, pt, False) >= 0 for pt in points)

# def get_valid_pothole_location(contour, mask_shape, pothole_size, filename):
#     """ Find a valid location where the full pothole fits inside the contour """
#     attempts = 100  # Retry limit
#     for _ in range(attempts):
#         x = random.randint(0, mask_shape[1] - pothole_size)
#         y = random.randint(0, mask_shape[0] - pothole_size)
#         x1, y1, x2, y2 = x, y, x + pothole_size, y + pothole_size

#         if is_pothole_inside_contour(contour, x1, y1, x2, y2):
#             return x1, y1, x2, y2  # Valid placement found

#     print(f"Warning: Could not find a valid placement inside the contour for {filename}")
#     return None  # If no valid location is found

# def poisson_blend_pothole(image, mask, pothole, filename):
#     """ Insert a pothole strictly inside the largest drivable area using Poisson blending """

#     # Convert mask to binary
#     _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

#     # Get the largest contour
#     largest_contour = get_largest_contour(binary_mask)
#     if largest_contour is None:
#         print(f"No valid drivable area found in the mask for {filename}.")
#         return image, None, None, None

#     # Randomly choose a size between 32x32 and 64x64
#     pothole_size = random.randint(32, 64)
#     pothole_resized = cv2.resize(pothole, (pothole_size, pothole_size))

#     # Find a fully valid location inside the contour
#     location = get_valid_pothole_location(largest_contour, mask.shape, pothole_size, filename)
#     if location is None:
#         return image, None, None, None

#     x1, y1, x2, y2 = location

#     # Ensure pothole fits inside image bounds
#     pothole_resized = pothole_resized[:(y2 - y1), :(x2 - x1)]

#     # Create pothole mask (white where pothole exists)
#     pothole_gray = cv2.cvtColor(pothole_resized, cv2.COLOR_BGR2GRAY)
#     _, pothole_mask = cv2.threshold(pothole_gray, 10, 255, cv2.THRESH_BINARY)

#     # Define center for Poisson blending
#     center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

#     # Perform seamless cloning (Poisson blending)
#     blended_image = cv2.seamlessClone(pothole_resized, image, pothole_mask, center, cv2.MIXED_CLONE)

#     # Convert bounding box coordinates to YOLO format (normalized)
#     img_h, img_w = image.shape[:2]
#     x_center = (x1 + x2) / 2 / img_w
#     y_center = (y1 + y2) / 2 / img_h
#     w_norm = (x2 - x1) / img_w
#     h_norm = (y2 - y1) / img_h

#     # YOLO annotation format: "class_id x_center y_center width height"
#     label_data = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"

#     return blended_image, label_data, largest_contour, pothole_size

# def visualize_labels(image, label_data, largest_contour, output_path):
#     """ 
#     Draw bounding boxes for each YOLO annotation and label them with the class name 
#     and bounding box size (width x height in pixels).
#     """
#     for line in label_data.strip().split("\n"):
#         if not line:
#             continue
#         values = line.split()
#         if len(values) < 5:
#             continue
#         # YOLO format: class_id x_center y_center width height
#         class_id, x_center, y_center, w, h = values
#         x_center, y_center, w, h = map(float, [x_center, y_center, w, h])
#         img_h, img_w = image.shape[:2]

#         # Convert normalized coordinates to pixel coordinates
#         x1 = int((x_center - w / 2) * img_w)
#         y1 = int((y_center - h / 2) * img_h)
#         x2 = int((x_center + w / 2) * img_w)
#         y2 = int((y_center + h / 2) * img_h)

#         # Compute bounding box size in pixels
#         width = x2 - x1
#         height = y2 - y1

#         # Define the label name based on class id (assume 0 corresponds to "Pothole")
#         label_name = "Pothole" if class_id == "0" else "Unknown"

#         # Draw the bounding box
#         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         # Overlay the label name and bounding box size
#         text = f"{label_name}: {width}x{height}"
#         cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#     if largest_contour is not None:
#         cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)

#     cv2.imwrite(output_path, image)

# def process_dataset(image_dir, mask_dir, pothole_dir, output_dir):
#     """ Process all images, blend 1-3 potholes using masks, save images, and generate labels """

#     image_output_dir = os.path.join(output_dir, "images")
#     label_output_dir = os.path.join(output_dir, "labels")
#     visualized_output_dir = os.path.join(output_dir, "visualized")

#     os.makedirs(image_output_dir, exist_ok=True)
#     os.makedirs(label_output_dir, exist_ok=True)
#     os.makedirs(visualized_output_dir, exist_ok=True)

#     image_files = sorted(os.listdir(image_dir))
#     pothole_files = sorted(os.listdir(pothole_dir))

#     for image_name in tqdm(image_files, desc="Processing images"):
#         if not image_name.lower().endswith(('.jpg', '.png')):
#             continue

#         image_path = os.path.join(image_dir, image_name)
#         mask_path = os.path.join(mask_dir, 'da_' + image_name)

#         if not os.path.exists(mask_path):
#             print(f"Mask not found for {image_name}, skipping.")
#             continue

#         # Read base image and mask
#         base_image = cv2.imread(image_path)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#         # Decide how many potholes to place (1 to 3)
#         num_potholes = random.randint(1, 3)

#         # Randomly sample distinct pothole images
#         if num_potholes > len(pothole_files):
#             # If you have fewer pothole images than num_potholes, 
#             # you might want to shuffle and reuse or just skip.
#             selected_potholes = pothole_files[:]  # fallback
#         else:
#             selected_potholes = random.sample(pothole_files, num_potholes)

#         # We'll blend sequentially on a copy of the base image
#         blended_image = base_image.copy()

#         # Collect all label data for this final image
#         combined_label_data = []

#         # For visualization contour purposes, we'll store the last valid contour
#         # (in practice, it doesn't change from the mask, but we'll keep it updated).
#         last_largest_contour = None

#         for pothole_name in selected_potholes:
#             pothole_path = os.path.join(pothole_dir, pothole_name)
#             pothole = cv2.imread(pothole_path, cv2.IMREAD_UNCHANGED)

#             # Convert to BGR if it has an alpha channel
#             if pothole.shape[-1] == 4:
#                 pothole = cv2.cvtColor(pothole, cv2.COLOR_BGRA2BGR)

#             # Blend the pothole onto the image
#             blended_image, label_data, largest_contour, _ = poisson_blend_pothole(
#                 blended_image, mask, pothole, image_name
#             )

#             # If we got a valid label (pothole was placed), store it
#             if label_data is not None:
#                 combined_label_data.append(label_data)
#                 last_largest_contour = largest_contour

#         # After placing up to 3 potholes, combine labels into one string
#         final_label_data = "".join(combined_label_data)

#         # If we have at least one pothole placed, save results
#         if final_label_data.strip():
#             # Save the blended image
#             output_image_path = os.path.join(image_output_dir, image_name)
#             cv2.imwrite(output_image_path, blended_image)

#             # Save the combined label file
#             label_filename = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
#             output_label_path = os.path.join(label_output_dir, label_filename)
#             with open(output_label_path, "w") as label_file:
#                 label_file.write(final_label_data)

#             # Visualize
#             visualize_labels(
#                 blended_image.copy(), 
#                 final_label_data, 
#                 last_largest_contour, 
#                 os.path.join(visualized_output_dir, image_name)
#             )

#     print(f"Processed images saved in {image_output_dir}")
#     print(f"Annotations saved in {label_output_dir}")
#     print(f"Visualized labels saved in {visualized_output_dir}")

# if __name__ == "__main__":
#     process_dataset(
#         image_dir="bdd_selected_250/images",
#         mask_dir="bdd_selected_250/mask",
#         pothole_dir="diffusion_model_generated_1000",
#         output_dir="outputs/bdd_diffusion_250"
#     )

import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def get_largest_contour(mask):
    """ Find the largest contour in the mask and return it """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # No valid region found

    # Sort contours by area and pick the largest one
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def is_pothole_inside_contour(contour, x1, y1, x2, y2):
    """ Check if the whole pothole fits inside the largest contour """
    points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]  # 4 corners
    return all(cv2.pointPolygonTest(contour, pt, False) >= 0 for pt in points)

def get_valid_pothole_location(contour, mask_shape, pothole_size, filename):
    """ Find a valid location where the full pothole fits inside the contour """
    attempts = 100  # Retry limit
    for _ in range(attempts):
        x = random.randint(0, mask_shape[1] - pothole_size)
        y = random.randint(0, mask_shape[0] - pothole_size)
        x1, y1, x2, y2 = x, y, x + pothole_size, y + pothole_size

        if is_pothole_inside_contour(contour, x1, y1, x2, y2):
            return x1, y1, x2, y2  # Valid placement found

    print(f"Warning: Could not find a valid placement inside the contour for {filename}")
    return None  # If no valid location is found

def poisson_blend_pothole(image, mask, pothole, filename):
    """ Insert a pothole strictly inside the largest drivable area using Poisson blending """

    # Convert mask to binary
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Get the largest contour
    largest_contour = get_largest_contour(binary_mask)
    if largest_contour is None:
        print(f"No valid drivable area found in the mask for {filename}.")
        return image, None, None, None

    # Randomly choose a size between 32x32 and 64x64
    pothole_size = random.randint(48, 64)
    pothole_resized = cv2.resize(pothole, (pothole_size, pothole_size))
    
    # Apply smoothing filters: median filtering and Gaussian blur to reduce pixelation
    pothole_resized = cv2.medianBlur(pothole_resized, 3)
    # pothole_resized = cv2.GaussianBlur(pothole_resized, (5, 5), 0)

    # Find a fully valid location inside the contour
    location = get_valid_pothole_location(largest_contour, mask.shape, pothole_size, filename)
    if location is None:
        return image, None, None, None

    x1, y1, x2, y2 = location

    # Ensure pothole fits inside image bounds
    pothole_resized = pothole_resized[:(y2 - y1), :(x2 - x1)]

    # Create pothole mask (white where pothole exists)
    pothole_gray = cv2.cvtColor(pothole_resized, cv2.COLOR_BGR2GRAY)
    _, pothole_mask = cv2.threshold(pothole_gray, 10, 255, cv2.THRESH_BINARY)

    # Define center for Poisson blending
    center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

    # Perform seamless cloning (Poisson blending)
    blended_image = cv2.seamlessClone(pothole_resized, image, pothole_mask, center, cv2.MIXED_CLONE)

    # Convert bounding box coordinates to YOLO format (normalized)
    # Loosen the bounding box by 10%
    padding_ratio = 0.4
    width = x2 - x1
    height = y2 - y1
    pad_w = int(width * padding_ratio / 2)
    pad_h = int(height * padding_ratio / 2)

    # Expand the box with padding and clip to image boundaries
    x1_padded = max(0, x1 - pad_w)
    y1_padded = max(0, y1 - pad_h)
    x2_padded = min(image.shape[1], x2 + pad_w)
    y2_padded = min(image.shape[0], y2 + pad_h)

    # Convert the padded bounding box to YOLO format (normalized)
    img_h, img_w = image.shape[:2]
    x_center = (x1_padded + x2_padded) / 2 / img_w
    y_center = (y1_padded + y2_padded) / 2 / img_h
    w_norm = (x2_padded - x1_padded) / img_w
    h_norm = (y2_padded - y1_padded) / img_h

    # YOLO annotation format: "class_id x_center y_center width height"
    label_data = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"

    return blended_image, label_data, largest_contour, pothole_size

def cutmix_pothole(image, mask, pothole, filename):
    """
    Insert a pothole strictly inside the largest drivable area using CutMix-style patch paste.
    """
    # Convert mask to binary
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Get the largest contour (drivable area)
    largest_contour = get_largest_contour(binary_mask)
    if largest_contour is None:
        print(f"No valid drivable area found in the mask for {filename}.")
        return image, None, None, None

    # Randomly choose a size between 32x32 and 64x64 for the pothole patch
    pothole_size = random.randint(48, 64)
    pothole_resized = cv2.resize(pothole, (pothole_size, pothole_size))

    # Find a valid random location inside the drivable contour
    location = get_valid_pothole_location(largest_contour, mask.shape, pothole_size, filename)
    if location is None:
        return image, None, None, None

    x1, y1, x2, y2 = location
    pothole_crop = pothole_resized[:(y2 - y1), :(x2 - x1)]

    # Perform direct paste (CutMix style)
    cutmix_image = image.copy()
    cutmix_image[y1:y2, x1:x2] = pothole_crop

    # Convert bounding box coordinates to YOLO format (normalized)
    # Loosen the bounding box by 10%
    padding_ratio = 0.1
    width = x2 - x1
    height = y2 - y1
    pad_w = int(width * padding_ratio / 2)
    pad_h = int(height * padding_ratio / 2)

    # Expand the box with padding and clip to image boundaries
    x1_padded = max(0, x1 - pad_w)
    y1_padded = max(0, y1 - pad_h)
    x2_padded = min(image.shape[1], x2 + pad_w)
    y2_padded = min(image.shape[0], y2 + pad_h)

    # Convert the padded bounding box to YOLO format (normalized)
    img_h, img_w = image.shape[:2]
    x_center = (x1_padded + x2_padded) / 2 / img_w
    y_center = (y1_padded + y2_padded) / 2 / img_h
    w_norm = (x2_padded - x1_padded) / img_w
    h_norm = (y2_padded - y1_padded) / img_h

    # YOLO annotation format: "class_id x_center y_center width height"
    label_data = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"

    return cutmix_image, label_data, largest_contour, pothole_size


def visualize_labels(image, label_data, largest_contour, output_path):
    """ 
    Draw bounding boxes for each YOLO annotation and label them with the class name 
    and bounding box size (width x height in pixels).
    """
    for line in label_data.strip().split("\n"):
        if not line:
            continue
        values = line.split()
        if len(values) < 5:
            continue
        # YOLO format: class_id x_center y_center width height
        class_id, x_center, y_center, w, h = values
        x_center, y_center, w, h = map(float, [x_center, y_center, w, h])
        img_h, img_w = image.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        x1 = int((x_center - w / 2) * img_w)
        y1 = int((y_center - h / 2) * img_h)
        x2 = int((x_center + w / 2) * img_w)
        y2 = int((y_center + h / 2) * img_h)

        # Compute bounding box size in pixels
        width = x2 - x1
        height = y2 - y1

        # Define the label name based on class id (assume 0 corresponds to "Pothole")
        label_name = "Pothole" if class_id == "0" else "Unknown"

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Overlay the label name and bounding box size
        text = f"{label_name}: {width}x{height}"
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if largest_contour is not None:
        cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)

# def process_dataset(image_dir, mask_dir, pothole_dir, output_dir):
#     """ Process all images, blend 1-3 potholes using masks, save images, and generate labels """

#     image_output_dir = os.path.join(output_dir, "images")
#     label_output_dir = os.path.join(output_dir, "labels")
#     visualized_output_dir = os.path.join(output_dir, "visualized")

#     os.makedirs(image_output_dir, exist_ok=True)
#     os.makedirs(label_output_dir, exist_ok=True)
#     os.makedirs(visualized_output_dir, exist_ok=True)

#     image_files = sorted(os.listdir(image_dir))
#     pothole_files = sorted(os.listdir(pothole_dir))

#     for image_name in tqdm(image_files, desc="Processing images"):
#         if not image_name.lower().endswith(('.jpg', '.png')):
#             continue

#         image_path = os.path.join(image_dir, image_name)
#         mask_path = os.path.join(mask_dir, 'da_' + image_name)

#         if not os.path.exists(mask_path):
#             print(f"Mask not found for {image_name}, skipping.")
#             continue

#         # Read base image and mask
#         base_image = cv2.imread(image_path)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#         # Decide how many potholes to place (1 to 3)
#         num_potholes = 1#random.randint(1, 3)

#         # Randomly sample distinct pothole images
#         if num_potholes > len(pothole_files):
#             selected_potholes = pothole_files[:]  # fallback if fewer pothole images exist
#         else:
#             selected_potholes = random.sample(pothole_files, num_potholes)

#         # Blend sequentially on a copy of the base image
#         blended_image = base_image.copy()
#         combined_label_data = []
#         last_largest_contour = None

#         for pothole_name in selected_potholes:
#             pothole_path = os.path.join(pothole_dir, pothole_name)
#             pothole = cv2.imread(pothole_path, cv2.IMREAD_UNCHANGED)

#             # Convert to BGR if it has an alpha channel
#             if pothole.shape[-1] == 4:
#                 pothole = cv2.cvtColor(pothole, cv2.COLOR_BGRA2BGR)

#             # Blend the pothole onto the image
#             # blended_image, label_data, largest_contour, _ = poisson_blend_pothole(
#             #     blended_image, mask, pothole, image_name
#             # )

#             blended_image, label_data, largest_contour, _ = cutmix_pothole(
#                 blended_image, mask, pothole, image_name
#             )

#             if label_data is not None:
#                 combined_label_data.append(label_data)
#                 last_largest_contour = largest_contour

#         final_label_data = "".join(combined_label_data)

#         if final_label_data.strip():
#             output_image_path = os.path.join(image_output_dir, image_name)
#             cv2.imwrite(output_image_path, blended_image)

#             label_filename = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
#             output_label_path = os.path.join(label_output_dir, label_filename)
#             with open(output_label_path, "w") as label_file:
#                 label_file.write(final_label_data)

#             visualize_labels(
#                 blended_image.copy(), 
#                 final_label_data, 
#                 last_largest_contour, 
#                 os.path.join(visualized_output_dir, image_name)
#             )

#     print(f"Processed images saved in {image_output_dir}")
#     print(f"Annotations saved in {label_output_dir}")
#     print(f"Visualized labels saved in {visualized_output_dir}")

def process_dataset(image_dir, mask_dir, pothole_dir, label_dir, output_dir, use_cutmix=True):
    """
    Processes the dataset, adds synthetic potholes, merges with existing labels if available.
    - use_cutmix=True for CutMix paste
    - use_cutmix=False for Poisson blending
    """

    image_output_dir = os.path.join(output_dir, "images")
    label_output_dir = os.path.join(output_dir, "labels")
    visualized_output_dir = os.path.join(output_dir, "visualized")

    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    os.makedirs(visualized_output_dir, exist_ok=True)

    image_files = sorted(os.listdir(image_dir))
    pothole_files = sorted(os.listdir(pothole_dir))

    for image_name in tqdm(image_files, desc="Processing images"):
        if not image_name.lower().endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, 'da_' + image_name)

        # Optional existing label path (may not exist for fresh datasets)
        label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt')) if label_dir else None

        if not os.path.exists(mask_path):
            print(f"Mask not found for {image_name}, skipping.")
            continue

        # Load image and mask
        base_image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Load existing labels if present
        existing_label_data = ""
        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as f:
                existing_label_data = f.read()

        if len(pothole_files) == 0:
            print("⚠️ No more synthetic potholes left to paste. Stopping.")
            break  # Stop processing if no potholes remain

        # Decide how many potholes to synthesize
        num_potholes = 1#random.randint(1)
        selected_potholes = random.sample(pothole_files, min(num_potholes, len(pothole_files)))

        pothole_files = [p for p in pothole_files if p not in selected_potholes]

        blended_image = base_image.copy()
        synthetic_label_data = ""
        last_largest_contour = None

        for pothole_name in selected_potholes:
            pothole_path = os.path.join(pothole_dir, pothole_name)
            pothole = cv2.imread(pothole_path, cv2.IMREAD_UNCHANGED)
            if pothole is None:
                print(f"❌ Failed to load pothole image: {pothole_path}")
                continue  # Skip this pothole and move on
            if pothole.shape[-1] == 4:
                pothole = cv2.cvtColor(pothole, cv2.COLOR_BGRA2BGR)

            # Apply CutMix or Poisson Blending based on flag
            if use_cutmix:
                blended_image, new_label, largest_contour, _ = cutmix_pothole(blended_image, mask, pothole, image_name)
            else:
                blended_image, new_label, largest_contour, _ = poisson_blend_pothole(blended_image, mask, pothole, image_name)

            if new_label:
                synthetic_label_data += new_label
                last_largest_contour = largest_contour

        # Merge original labels (if any) with new synthetic labels
        combined_labels = existing_label_data.strip() + "\n" + synthetic_label_data.strip()
        combined_labels = combined_labels.strip() + "\n"

        # Save blended image and combined labels
        output_image_path = os.path.join(image_output_dir, image_name)
        cv2.imwrite(output_image_path, blended_image)

        output_label_path = os.path.join(label_output_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(output_label_path, "w") as label_file:
            label_file.write(combined_labels)

        # Optional visualization
        visualize_labels(
            blended_image.copy(),
            combined_labels,
            last_largest_contour,
            os.path.join(visualized_output_dir, image_name)
        )

    print(f"✅ Processed images saved in {image_output_dir}")
    print(f"✅ Updated annotations saved in {label_output_dir}")
    print(f"✅ Visualized labels saved in {visualized_output_dir}")

if __name__ == "__main__":
    process_dataset(
        image_dir="dataset_japan_no_pothole_540/images",
        mask_dir="dataset_japan_no_pothole_540/mask",
        pothole_dir="potholes/diffusion_filtered_0.26_540",
        label_dir=None,  # Set to None if it's a fresh dataset without labels
        output_dir="outputs/japan_diffusion_540_filtered_polygon",
        use_cutmix=False  # Toggle between CutMix or Poisson
    )