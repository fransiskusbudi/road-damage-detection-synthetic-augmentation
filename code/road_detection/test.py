import os
import cv2
import torch
import argparse
import onnxruntime as ort
import numpy as np

def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w, h
    new_unpad_w, new_unpad_h = new_unpad
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h

    dw, dh = pad_w // 2, pad_h // 2  # divide padding into 2 sides

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

    return canvas, r, dw, dh, new_unpad_w, new_unpad_h

def infer_drive_area(weight, img_dir, output_dir):
    ort.set_default_logger_severity(4)
    onnx_path = weight
    ort_session = ort.InferenceSession(onnx_path)
    print(f"Loaded {onnx_path} successfully!")

    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue  # Skip non-image files

        img_path = os.path.join(img_dir, img_name)
        print(f"Processing: {img_path}")

        img_bgr = cv2.imread(img_path)
        height, width, _ = img_bgr.shape

        img_rgb = img_bgr[:, :, ::-1].copy()

        # Preprocess
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))

        img = canvas.copy().astype(np.float32) / 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225

        img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]  # (1, 3, 640, 640)

        # Run inference (only drive area segmentation)
        da_seg_out = ort_session.run(
            ['drive_area_seg'],
            input_feed={"images": img}
        )[0]

        # Process drive area segmentation
        da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
        da_seg_mask = np.argmax(da_seg_out, axis=1)[0] * 255
        da_seg_mask = cv2.resize(da_seg_mask.astype(np.uint8), (width, height))

        # Save drive area segmentation output
        save_da_path = os.path.join(output_dir, f"da_{img_name}")
        cv2.imwrite(save_da_path, da_seg_mask)

        print(f"Saved drive area segmentation: {save_da_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, required=True, help="Path to YOLOP ONNX model")
    parser.add_argument('--img_dir', type=str, required=True, help="Directory of input images")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save inference results")

    args = parser.parse_args()
    infer_drive_area(weight=args.weight, img_dir=args.img_dir, output_dir=args.output_dir)
