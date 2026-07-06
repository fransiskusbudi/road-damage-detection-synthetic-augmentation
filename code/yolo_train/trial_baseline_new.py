import time
import torch
from ultralytics import YOLO
import os
import pandas as pd  # Import pandas for saving results as CSV

def main():
    batch_sizes = [16]
    learning_rates = [0.001]
    lrf1 = 0.01
    weight_decay= 0.005
    
    model_names = ["yolov8n"]
    # datasets = ['baseline_17Mar','bdd_diffusion_108_filtered_0.26','bdd_diffusion_215_filtered_0.26','bdd_diffusion_323_filtered_0.26','bdd_diffusion_430_filtered_0.26',
    # 'bdd_wgan_108_filtered_0.28','bdd_wgan_215_filtered_0.28','bdd_wgan_323_filtered_0.28','bdd_wgan_430_filtered_0.28']
    # datasets = ['japan_diffusion_540_filtered_0.26']
    datasets = ['albumentations_100']
    base_path = "/disk/scratch/USER/mlpractical_rdd/diffusion_gan/"
    optimizer = "AdamW"
    
    ###### Check if CUDA is available and set the device accordingly
    device_num = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 512
    
    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset)
        yaml_file = os.path.join(dataset_path, "dataset.yaml")
        
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for model_name in model_names:
                    name = f"{dataset}_{optimizer}_{batch_size}_lr_{learning_rate}_{lrf1}_{model_name}_{image_size}"
                    start_time = time.time()
                    
                    print("=" * 72)
                    print(f"Training on {dataset} with {yaml_file}, BatchSize={batch_size}, LearningRate={learning_rate}, Model={model_name}, Name={name}")
                    print("=" * 72)
                    
                    # ✅ Ensure the model file exists
                    model_path = f"{model_name}.pt"
                    if not os.path.exists(model_path):
                        print(f"❌ ERROR: Model file '{model_path}' not found!")
                        return
                    
                    # Load YOLO model
                    model = YOLO(model_path)
                    
                    # ===================== TRAINING =====================
                    output_folder = f"outputs/{dataset}/{name}/"
                    train_folder = os.path.join(output_folder)  # ✅ Separate training folder
                    os.makedirs(train_folder, exist_ok=True)
                    
                    results = model.train(
                        data=yaml_file,
                        epochs=100,
                        imgsz=image_size,
                        device=device_num,
                        batch=batch_size,
                        project=train_folder,  # ✅ Save training results in outputs/{dataset}/{name}/train/
                        optimizer=optimizer,
                        lr0=learning_rate,
                        lrf=lrf1,
                        weight_decay=weight_decay,
                        save_period=10,
                        cos_lr=True,
                        workers=4,
                        mixup = 0.2,
                        shear = 0.2,
                        copy_paste=0.1
                        # ,hsv_h= 0.0,  # hue
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
                        # mixup= False  # mixup
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
                    print(f"Evaluating {model_name} on {dataset} test dataset")
                    print("=" * 72)
                    
                    test_folder = os.path.join(output_folder, "test")  # ✅ Separate testing folder
                    os.makedirs(test_folder, exist_ok=True)
                    
                    # Evaluate on the test dataset
                    test_results = model.val(
                        data=yaml_file,
                        split="test",
                        imgsz=image_size,
                        device=device_num,
                        project=test_folder,  # ✅ Save test results in outputs/{dataset}/{name}/test/
                    )
                    
                    print("#" * 69)
                    print(f"Test Evaluation Results: {test_results}")
                    print("#" * 69)
                    
                    # ===================== SAVE TEST RESULTS TO CSV =====================
                    csv_filename = os.path.join(test_folder, f"test_results_{model_name}.csv")
                    
                    # Extract relevant metrics
                    metrics = {
                        "Dataset": dataset,
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
                    
                    # ✅ Save results to `outputs/{dataset}/{name}/test/`
                    df.to_csv(csv_filename, index=False)
                    print(f"✅ Test results saved inside: {csv_filename}")

if __name__ == "__main__":
    main()
