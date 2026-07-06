import time
import torch
from ultralytics import YOLO
import os
import yaml
import argparse

def main():
    batch_sizes = [256]
    learning_rates = [0.01]
    lrf1 = 0.01

    model_names = ["yolov8n"]
    yaml_file = 'dataset.yaml'
    optimizer = 'auto'
    ###### Check if CUDA is available and set the device accordingly
    # if torch.cuda.is_available():
    #     device_num = [0]  # Use GPU 0 if available
    # else:
    #     device_num = ['cpu']  # Fallback to CPU if GPU is not available
    device_num = [0,1,2,3]
    image_size = 384
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for model_name in model_names:
                name = f"{optimizer}_{batch_size}_lr_{learning_rate}_{lrf1}_{model_name}_{image_size}"
                start_time = time.time()
                print("=" * 72)
                print(f"Training with {yaml_file}, BatchSize={batch_size}, LearningRate={learning_rate}, Model={model_name}, Name={name}")
                print("=" * 72)
                model = YOLO(f'{model_name}.pt')
                results = model.train(
                    data=f'{yaml_file}',
                    epochs=100,
                    imgsz=image_size,
                    device=device_num,
                    batch=batch_size,
                    name=name,
                    optimizer=optimizer,
                    save_period = 5,
                    cos_lr=True,
                    patience=0,
                    workers = 4
                    # ,
                    # hsv_h= 0.0,  # hue
                    # hsv_s= 0.0,  # saturation
                    # hsv_v= 0.0,  # value
                    # degrees= 0.0,  # rotation
                    # translate= 0.0,  # translate
                    # scale= 0.0,  # scale
                    # shear= 0.0,  # shear
                    # perspective= 0.0,  # perspective
                    # flipud= 0.0,  # flip up-down
                    # fliplr= 0.0,  # flip left-right
                    # mosaic= False,  # mosaic
                    # mixup= False,  # mixup
                    # erasing = 0.0
                )

                end_time = time.time()
                elapsed_time = end_time - start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)

                print("#" * 69)
                print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
                print("#" * 69)
                    
if __name__ == "__main__":
    main()
