import time
import torch
from ultralytics import YOLO
import os
import yaml
import argparse
import pandas as pd  # Import pandas for saving results as CSV

def main():
    print('start')
    batch_sizes = [16]
    learning_rates = [0.001]
    lrf1 = 0.2
    weight_decay= 0.01

    model_names = ["yolov8m"]
    yaml_file = 'dataset.yaml'
    optimizer = 'AdamW'
    
    ###### Check if CUDA is available and set the device accordingly
    # if torch.cuda.is_available():
    #     device_num = [0]  # Use GPU 0 if available
    # else:
    #     device_num = ['cpu']  # Fallback to CPU if GPU is not available
    device_num = [0,1]
    image_size = 384
    
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for model_name in model_names:
                name = f"bdd_diffusion100_{optimizer}_{batch_size}_lr{learning_rate}_{lrf1}_{model_name}_{image_size}"
                start_time = time.time()
                
                print("=" * 72)
                print(f"Training with {yaml_file}, BatchSize={batch_size}, LearningRate={learning_rate}, Model={model_name}, Name={name}")
                print("=" * 72)
                
                # Load YOLO model
                model = YOLO(f'{model_name}.pt')

                # Train model
                results = model.train(
                    data=f'{yaml_file}',
                    epochs=100,
                    imgsz=image_size,
                    device=device_num,
                    batch=batch_size,
                    lr0=learning_rate,
                    lrf=lrf1,
                    weight_decay=weight_decay,
                    name=name,
                    optimizer=optimizer,
                    save_period=5,
                    cos_lr=True,
                    patience=0,
                    workers=4
                )

                end_time = time.time()
                elapsed_time = end_time - start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)

                print("#" * 69)
                print(f"Training completed in: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
                print("#" * 69)

                # ===================== TESTING =====================
                print("=" * 72)
                print(f"Evaluating {model_name} on the test dataset")
                print("=" * 72)

                # Determine training folder path dynamically
                training_folder = f"runs/detect/{name}/"
                os.makedirs(training_folder, exist_ok=True)  # Ensure directory exists

                # Evaluate on the test dataset
                test_results = model.val(
                    data=f'{yaml_file}',  # Ensure dataset.yaml has a 'test' split
                    split="test",  # Specify the test dataset
                    imgsz=image_size,
                    device=device_num
                )

                print("#" * 69)
                print(f"Test Evaluation Results: {test_results}")
                print("#" * 69)

                # ===================== SAVE TEST RESULTS TO CSV =====================
                csv_filename = os.path.join(training_folder, f"test_results_{model_name}.csv")

                # Extract relevant metrics
                metrics = {
                    "Model": model_name,
                    "Batch Size": batch_size,
                    "Learning Rate": learning_rate,
                    "mAP_50": test_results.box.map50,   # mAP@50
                    "mAP_50_95": test_results.box.map,  # mAP@50:95
                    "Precision": test_results.box.p,    # Precision
                    "Recall": test_results.box.r,       # Recall
                    "F1_score": test_results.box.f1     # F1 Score
                }

                # Convert to DataFrame
                df = pd.DataFrame([metrics])

                # Save results to CSV inside the same training folder
                df.to_csv(csv_filename, index=False)
                print(f"Test results saved inside training folder: {csv_filename}")

if __name__ == "__main__":
    main()