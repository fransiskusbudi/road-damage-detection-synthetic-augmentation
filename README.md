# Beyond GANs: A Comparative Study of Generative Models and Traditional Augmentation Techniques for Pothole Detection

MSc group project (G014). The project studies whether **synthetically generated pothole images**
can augment a real dataset and improve a **YOLO** road-damage detector. Several generative
approaches are compared, their outputs are filtered for realism with **CLIP**, blended into
real road scenes, and used to train and evaluate detection performance.

## Summary

A **YOLOv8** pothole detector is trained on the **Road Damage Dataset (RDD2022)** and augmented
with synthetic potholes from three generative approaches — **diffusion models**, **WGAN-GP**, and
**SinGAN** — as well as **traditional augmentation** (rotation, flips, colour, noise). Synthetic
potholes are blended into real "empty" road scenes with **Poisson blending** and filtered for
realism using **CLIP**. The central question: do generative models justify their added complexity
over simple augmentation, especially for **cross-country generalization**?

**Key findings:**

- **Diffusion @ 30% synthetic** gives the best **mAP@0.5 = 0.322**, a **+2.9%** improvement over the
  0.293 baseline; diffusion @ 20% gives the best **recall (0.347)**.
- **WGAN-GP** yields the highest **precision (0.610)** — fewer false positives, but lower recall (0.300).
- **Generative models beat traditional augmentation only narrowly** (0.316 vs 0.322 mAP@0.5) — simple
  augmentations remain highly competitive.
- **SinGAN underperforms** consistently; its single-image approach lacks the sample diversity of the
  full-dataset methods.

Overall, synthetic data — diffusion in particular — measurably improves cross-country generalization,
but the margin over traditional augmentation is small.

## Pipeline

```
generate potholes  ──►  filter by realism  ──►  paste into road scenes  ──►  train + evaluate
(diffusion / WGAN /      (CLIP scoring:            (Poisson blending)          (YOLO on RDD,
 SinGAN)                  "A real road pothole")                                class D40 = pothole)
```

## Components

| Folder | Description | Author |
|--------|-------------|--------|
| `code/yolo_train/` | Trains and evaluates the YOLO detector on the RDD dataset (class `D40` = pothole), across baseline and augmented data mixes. | **This work** |
| `code/diffusion_model/` | DDPM (HuggingFace `diffusers`) trained to synthesize pothole images; sampling and metric scripts. | **This work** |
| `code/pothole_pasting/` | Poisson-blends generated potholes into real road images and scores realism with OpenCLIP to filter samples. | **This work** |
| `code/wgan/` | WGAN-GP baseline for pothole generation. | Based on [EmilienDupont/wgan-gp](https://github.com/EmilienDupont/wgan-gp) (MIT) |
| `code/singan/` | SinGAN single-image generative baseline. Used near-stock (one training tweak). | Based on [tamarott/SinGAN](https://github.com/tamarott/SinGAN) (MIT) |
| `code/road_detection/` | YOLOP panoptic road detection, used for road-surface segmentation. | Based on [hustvl/YOLOP](https://github.com/hustvl/YOLOP) (MIT) |

Root notebooks/figures (`collate.ipynb`, `comparison_grid.png`, `1–4.jpeg`) collate and
visualise results across the generative methods.

## Attribution

`code/wgan`, `code/singan`, and `code/road_detection` are third-party open-source projects
included here so the experiments are reproducible. Each retains its original `LICENSE` and
copyright notice. All credit for those components goes to their respective authors; only the
pothole-specific modifications and integration are part of this work.

## Notes

- Scripts contain absolute paths from the university compute cluster (e.g. dataset and
  pretrained-weight locations) — update these to your own paths before running.
- Model weights, datasets, and generated outputs are not included in this repository.
