from PIL import Image
import os
import math

# Define the folder path where fake images are stored
fake_images_folder = "fake_images_v2"  # Adjust this path if needed
output_image_path = "fake_images_v2/collated_image.png"

# Load all images from the folder
image_files = [os.path.join(fake_images_folder, f) for f in os.listdir(fake_images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # Ensure the order remains consistent

# Load images into a list
images = [Image.open(img) for img in image_files]

# Get image dimensions (assuming all images have the same size)
img_width, img_height = images[0].size

# Define grid size (adjust automatically for 10 images)
num_images = len(images)
cols = math.ceil(math.sqrt(num_images))  # Number of columns
rows = math.ceil(num_images / cols)  # Number of rows

# Create a blank canvas for the final image
collated_width = cols * img_width
collated_height = rows * img_height
collated_image = Image.new("RGB", (collated_width, collated_height))

# Paste images into the grid
for index, img in enumerate(images):
    x_offset = (index % cols) * img_width
    y_offset = (index // cols) * img_height
    collated_image.paste(img, (x_offset, y_offset))

# Save the final collated image
collated_image.save(output_image_path)
print(f"Collated image saved at {output_image_path}")
