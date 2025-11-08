# WellView

WellView is a human-centric 3D computational framework for optimizing window views in building design.  
It integrates photogrammetric 3D city models, virtual photography in Blender, and deep learning–based semantic segmentation (Mask2Former) to evaluate and optimize window configurations.

The original experiments were based on **Google 3D Tiles**. Due to licensing and copyright constraints, the raw 3D Tiles data cannot be redistributed in this repository.  
Instead, we provide a cleaned **Blender scene (.blend)** derived from the 3D Tiles for research use.  
If you would like access to the Blender model for academic or research purposes, please request it via the following link:  
👉 [Google Drive link](https://drive.google.com/file/d/1YoJsiHbadIuwfQCRdhkitDqAiZSY07fe/view?usp=sharing)

<br>

<p align="center">
  <img width="2000" height="1125" alt="image" src="https://github.com/user-attachments/assets/3b6a4aae-05ba-4c4e-8957-e32a15302f87" />
  <img width="2000" height="1125" alt="image" src="https://github.com/user-attachments/assets/b7c39701-9a5b-424a-9c51-aa20995fb68e" />
  <img width="2000" height="1125" alt="image" src="https://github.com/user-attachments/assets/81aa4268-ba1a-4639-a5a8-f9b8b77bced8" />
</p>

---

## Features

- **3D urban context**
  - Uses photogrammetric / 3D Tiles data imported into **Blender** to reconstruct realistic urban environments.

- **Parametric window design**
  - Window position *(x, y)* and size *(width, height)* are controlled via Blender scripts.
  - Supports different design configurations: **monolithic**, **grouped**, and **individual** window optimization.

- **Virtual photography**
  - Multiple interior viewpoints are sampled per room to simulate what occupants actually see through the windows.

  <p align="center">
    <img width="880" height="367" alt="viewpoints" src="https://github.com/user-attachments/assets/a20a2a49-fcd5-432c-afa9-61dbaa699da5" />
  </p>

- **Semantic segmentation**
  - Rendered views are parsed with **Mask2Former** (via **Detectron2**) to classify pixels into semantic classes such as *building, sky, greenery, water*.

  <p align="center">
    <img width="618" height="390" alt="segmentation" src="https://github.com/user-attachments/assets/e0d74084-9e31-4da0-8c31-fa375a2e8c40" />
  </p>

- **Quantitative view index (WVQS)**
  - Computes a Window View Quality Score (WVQS) based on pixel-level semantic composition, balancing positive elements (e.g., greenery, sky, water) and negative elements (e.g., buildings).

- **Optimization**
  - Supports algorithmic search (e.g., **grid search**, **Bayesian optimization**) to find window configurations that maximize WVQS.

---

## Environment & Dependencies

This repository assumes the following software environment:

- **Blender**: 4.3 (or later) – used to author 3D scenes and render images.
- **Python**: e.g., 3.9 / 3.10 (adjust to your local setup).
- **PyTorch**: for deep learning and Mask2Former inference.
- **Detectron2**: framework used to run Mask2Former.
- **Mask2Former**: semantic segmentation model for parsing window views.

Minimal dependency list (to be adapted to your actual environment):

```text
Blender 4.3
Python >= 3.9
torch
torchvision
detectron2
opencv-python
numpy
matplotlib
