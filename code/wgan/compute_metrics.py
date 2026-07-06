import torch_fidelity

# Define the paths to your datasets
real_images = "cropped_400"
generated_images = {
    "wgan_filtered_generated_images_1000_threshold_0.28": "wgan_filtered_generated_images_1000_threshold_0.28"
}

# Compute FID and IS for each generated set
results = {}
for name, path in generated_images.items():
    metrics = torch_fidelity.calculate_metrics(
        input1=path,
        input2=real_images,
        isc=True,  # Inception Score
        fid=True   # Fréchet Inception Distance
    )
    results[name] = metrics
    print(f"Results for {name}: {metrics}")

# Print final results
print("\nFinal FID & IS Scores:")
for name, metrics in results.items():
    print(f"{name}: {metrics}")
