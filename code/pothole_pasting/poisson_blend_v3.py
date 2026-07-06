
import os
import cv2
import numpy as np
import random
from tqdm import tqdm

from skimage.exposure import match_histograms

def histogram_match_pothole(pothole, background_patch):
    """Match the pothole's histogram to the sampled background patch"""
    pothole_matched = match_histograms(pothole, background_patch, channel_axis=-1)
    return np.clip(pothole_matched, 0, 255).astype(np.uint8)

def add_asphalt_noise(image, strength=0.4):
    """
    Adds random high-frequency noise to simulate asphalt texture.
    Helps the model pick up texture features after Poisson smoothing.
    """
    noise = np.random.normal(0, 25, image.shape).astype(np.float32)
    noisy_image = cv2.addWeighted(image.astype(np.float32), 1.0, noise, strength, 0)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

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

def match_pixel_grain(pothole_patch):
    """
    Adjust pothole patch graininess by downscaling and upscaling to match background pixel grain.
    """
    scale_factor = 0.9  # Control how much detail you drop
    h, w = pothole_patch.shape[:2]
    downscaled = cv2.resize(pothole_patch, (int(w * scale_factor), int(h * scale_factor)))
    upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
    return upscaled

def apply_gamma_correction(image, gamma=1.2):
    """
    Apply gamma correction for subtle brightness and contrast boost.
    """
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def perspective_warp(patch):
    """ Apply random perspective warp to the pothole patch """
    h, w = patch.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    warp_strength = 10  # Max pixel shift for corners
    dst_pts = np.float32([
        [random.randint(0, warp_strength), random.randint(0, warp_strength)],
        [w - random.randint(0, warp_strength), random.randint(0, warp_strength)],
        [w - random.randint(0, warp_strength), h - random.randint(0, warp_strength)],
        [random.randint(0, warp_strength), h - random.randint(0, warp_strength)]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_patch = cv2.warpPerspective(patch, M, (w, h))
    return warped_patch

def edge_mask(pothole_patch):
    gray = cv2.cvtColor(pothole_patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Tune thresholds
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create blank mask and draw only the pothole shape
    mask = np.zeros_like(gray)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    
    return mask

def kmeans_mask(pothole_patch, k=2):
    reshaped = pothole_patch.reshape((-1, 3))
    reshaped = np.float32(reshaped)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reshape labels into mask
    clustered = labels.reshape((pothole_patch.shape[:2]))
    mask = np.where(clustered == 1, 255, 0).astype(np.uint8)  # Tune based on clustering

    return mask


def grabcut_pothole_mask(pothole_patch):
    mask = np.zeros(pothole_patch.shape[:2], np.uint8)
    
    # Initialize background and foreground models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define an initial rectangle around the pothole
    h, w = pothole_patch.shape[:2]
    rect = (5, 5, w-10, h-10)  # Shrink to avoid border effects

    cv2.grabCut(pothole_patch, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    # Convert GrabCut mask to binary (foreground = 1, background = 0)
    mask_final = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

    return mask_final

def hybrid_pothole_mask(pothole_patch):
    
    # Step 2: Edge detection to refine boundaries
    mask_edge = edge_mask(pothole_patch)

    # Step 3: Adaptive threshold to capture fine details
    gray = cv2.cvtColor(pothole_patch, cv2.COLOR_BGR2GRAY)
    mask_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2
    )

    # Step 4: Combine all masks
    final_mask = cv2.bitwise_or(mask_adaptive)

    return final_mask

def create_pothole_mask(pothole_patch):
    pothole_gray = cv2.cvtColor(pothole_patch, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold for better edge detection
    pothole_mask = cv2.adaptiveThreshold(
        pothole_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=5, C=2
    )
    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    pothole_mask = cv2.morphologyEx(pothole_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    pothole_mask = cv2.morphologyEx(pothole_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Keep largest contour only
    contours, _ = cv2.findContours(pothole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_clean = np.zeros_like(pothole_mask)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_clean, [largest_contour], -1, 255, -1)
    # Feather edges
    pothole_mask = cv2.GaussianBlur(mask_clean, (11, 11), sigmaX=5)
    return pothole_mask

def poisson_blend_pothole(image, mask, pothole, filename):
    """ Insert a pothole strictly inside the largest drivable area using Poisson blending """

    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    largest_contour = get_largest_contour(binary_mask)
    if largest_contour is None:
        print(f"No valid drivable area found in the mask for {filename}.")
        return image, None, None, None

    pothole_size = random.randint(64, 128)
    padding_ratio = 0.2
    pothole_mask = grabcut_pothole_mask(pothole)
    kernel = np.ones((7, 7), np.uint8)
    pothole_mask = cv2.dilate(pothole_mask, kernel, iterations=4)
    # pothole_mask = cv2.GaussianBlur(pothole_mask, (11, 11), sigmaX=5)


    if is_mask_mostly_black(pothole_mask, threshold=0.7):
        print(f"Skipping {filename}: pothole mask mostly black")
        return image, None, None, None

    pothole = cv2.GaussianBlur(pothole, (5, 5), sigmaX=1)
    pothole_resized = cv2.resize(pothole, (pothole_size, pothole_size))
    
    pothole_mask = cv2.resize(pothole_mask, (pothole_size, pothole_size), interpolation=cv2.INTER_NEAREST)

    location = get_valid_pothole_location(largest_contour, mask.shape, pothole_size, filename)
    if location is None:
        return image, None, None, None

    x1, y1, x2, y2 = location
    pothole_resized = pothole_resized[:(y2 - y1), :(x2 - x1)]

    # ✅ Pothole mask creation
    # pothole_gray = cv2.cvtColor(pothole_resized, cv2.COLOR_BGR2GRAY)
    # pothole_gray = cv2.equalizeHist(pothole_gray)
    # _, pothole_mask = cv2.threshold(pothole_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # pothole_mask = cv2.GaussianBlur(pothole_mask, (11, 11), sigmaX=5)

    


    # ✅ Define center for Poisson blending
    center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

    # ✅ Perform seamless cloning with histogram-matched pothole
    blended_image = cv2.seamlessClone(pothole_resized, image, pothole_mask, center, cv2.MIXED_CLONE)
    # blended_image = cv2.cvtColor(blended_image, cv2.COLOR_BGR2GRAY)
    
    # ✅ YOLO bounding box calculation with padding
    width = x2 - x1
    height = y2 - y1
    pad_w = int(width * padding_ratio / 2)
    pad_h = int(height * padding_ratio / 2)
    x1_padded = max(0, x1 - pad_w)
    y1_padded = max(0, y1 - pad_h)
    x2_padded = min(image.shape[1], x2 + pad_w)
    y2_padded = min(image.shape[0], y2 + pad_h)

    img_h, img_w = image.shape[:2]
    x_center = (x1_padded + x2_padded) / 2 / img_w
    y_center = (y1_padded + y2_padded) / 2 / img_h
    w_norm = (x2_padded - x1_padded) / img_w
    h_norm = (y2_padded - y1_padded) / img_h
    label_data = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"

    return blended_image, label_data, largest_contour, pothole_size

def is_mask_mostly_black(mask, threshold=0.90):
    """
    Returns True if the mask is mostly black (>= threshold percentage of black pixels)
    """
    total_pixels = mask.size
    black_pixels = np.sum(mask == 0)
    return (black_pixels / total_pixels) >= threshold

def enlarge_mask(mask, scale=1.5):
    """Enlarge the binary mask by the given scale while keeping it centered"""
    h, w = mask.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    enlarged_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Paste enlarged mask back into original size canvas, centered
    final_mask = np.zeros_like(mask)
    x_offset = max((final_mask.shape[1] - enlarged_mask.shape[1]) // 2, 0)
    y_offset = max((final_mask.shape[0] - enlarged_mask.shape[0]) // 2, 0)

    # Compute valid region to paste
    y_end = min(y_offset + enlarged_mask.shape[0], final_mask.shape[0])
    x_end = min(x_offset + enlarged_mask.shape[1], final_mask.shape[1])
    final_mask[y_offset:y_end, x_offset:x_end] = enlarged_mask[0:(y_end - y_offset), 0:(x_end - x_offset)]

    return final_mask

def blend_with_background_texture(pothole_crop, background_patch, alpha=0.3):
    """
    Blend some background texture into the pothole patch to reduce texture mismatch.
    Alpha controls how much of the background is injected into the pothole.
    """
    if background_patch.shape != pothole_crop.shape:
        background_patch = cv2.resize(background_patch, (pothole_crop.shape[1], pothole_crop.shape[0]))
    blended = cv2.addWeighted(pothole_crop.astype(np.float32), 1 - alpha, background_patch.astype(np.float32), alpha, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)

def cutmix_pothole(image, mask, pothole, filename):
    """
    Insert a pothole strictly inside the largest drivable area using CutMix-style patch paste with GrabCut mask.
    Mask is generated first (full resolution), then both pothole & mask are resized together.
    """
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    largest_contour = get_largest_contour(binary_mask)
    if largest_contour is None:
        print(f"No valid drivable area found in the mask for {filename}.")
        return image, None, None, None

    # ✅ Generate GrabCut mask at full resolution first
    pothole_mask_full = grabcut_pothole_mask(pothole)
    kernel = np.ones((5, 5), np.uint8)
    pothole_mask_full = cv2.dilate(pothole_mask_full, kernel, iterations=4)

    if is_mask_mostly_black(pothole_mask_full):
        print(f"Skipping {filename}: pothole mask mostly black")
        return image, None, None, None

    # ✅ Randomly choose pothole size and resize both pothole and mask together
    pothole_size = random.randint(48, 64)
    pothole_resized = cv2.resize(pothole, (pothole_size, pothole_size))
    pothole_mask_resized = cv2.resize(pothole_mask_full, (pothole_size, pothole_size), interpolation=cv2.INTER_NEAREST)
    

    # pothole_resized = add_asphalt_noise(pothole_resized, strength=0.3)
    # pothole_mask_resized = enlarge_mask(pothole_mask_resized, scale=1.)

    # ✅ Find valid location inside the drivable contour
    location = get_valid_pothole_location(largest_contour, mask.shape, pothole_size, filename)
    if location is None:
        return image, None, None, None

    x1, y1, x2, y2 = location
    crop_h, crop_w = (y2 - y1), (x2 - x1)
    if crop_h <= 0 or crop_w <= 0:
        print(f"Invalid crop size in {filename}")
        return image, None, None, None

    # ✅ Crop pothole and mask properly
    pothole_crop = pothole_resized[:crop_h, :crop_w]
    mask_crop = pothole_mask_resized[:crop_h, :crop_w]

    # ✅ 3-channel mask for CutMix
    mask_3ch = cv2.merge([mask_crop, mask_crop, mask_crop])


    background_patch = image[y1:y2, x1:x2]

    # ✅ Blend some background texture into pothole
    pothole_crop = blend_with_background_texture(pothole_crop, background_patch, alpha=0.6)

    # ✅ CutMix paste
    cutmix_image = image.copy()
    roi = cutmix_image[y1:y2, x1:x2]
    cutmix_result = np.where(mask_3ch == 255, pothole_crop, roi)

        # ✅ Sample background texture patch
    cutmix_image[y1:y2, x1:x2] = cutmix_result

    # ✅ YOLO bounding box with padding
    padding_ratio = 0.2
    width, height = (x2 - x1), (y2 - y1)
    pad_w, pad_h = int(width * padding_ratio / 2), int(height * padding_ratio / 2)

    x1_padded = max(0, x1 - pad_w)
    y1_padded = max(0, y1 - pad_h)
    x2_padded = min(image.shape[1], x2 + pad_w)
    y2_padded = min(image.shape[0], y2 + pad_h)

    img_h, img_w = image.shape[:2]
    x_center = (x1_padded + x2_padded) / 2 / img_w
    y_center = (y1_padded + y2_padded) / 2 / img_h
    w_norm = (x2_padded - x1_padded) / img_w
    h_norm = (y2_padded - y1_padded) / img_h

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
        if not synthetic_label_data.strip():
            print(f"⚠️ Skipping {image_name}: No valid pothole inserted (mask likely too black).")
            continue

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
        image_dir="bdd_selected_540/images",
        mask_dir="bdd_selected_540/mask",
        pothole_dir="potholes/diffusion_filtered_0.26_540",
        label_dir=None,  # Set to None if it's a fresh dataset without labels
        output_dir="outputs/trial_bdd_540",
        use_cutmix=False  # Toggle between CutMix or Poisson
    )