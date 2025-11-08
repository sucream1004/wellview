# WellView

WellView is a human-centric 3D computational framework for optimizing window views in building design.  
It integrates photogrammetric 3D city models, virtual photography in Blender, and deep learning–based semantic segmentation (Mask2Former) to evaluate and optimize window configurations.

<br>

<p align="center">
  <img width="2000" height="1125" alt="image" src="https://github.com/user-attachments/assets/3b6a4aae-05ba-4c4e-8957-e32a15302f87" />
<img width="2000" height="1125" alt="image" src="https://github.com/user-attachments/assets/b7c39701-9a5b-424a-9c51-aa20995fb68e" />
<img width="2000" height="1125" alt="image" src="https://github.com/user-attachments/assets/81aa4268-ba1a-4639-a5a8-f9b8b77bced8" />

</p>

---

## Features

- **3D urban context**
  - Uses photogrammetric / 3D tiles data imported into **Blender** for realistic urban environments.
- **Parametric window design**
  - Window position (x, y) and size (width, height) are controlled via scripts in Blender.
  - Supports different design configurations (monolithic, grouped, individual windows).
- **Virtual photography**
  - Multiple interior viewpoints are sampled per room to simulate what occupants see through the windows.
- **Semantic segmentation**
  - Rendered views are parsed with **Mask2Former** (via **Detectron2**) to classify pixels into classes such as *building, sky, greenery, water*.
- **Quantitative view index (WVQS)**
  - Computes a Window View Quality Score (WVQS) based on pixel-level semantic composition.
- **Optimization**
  - Supports algorithmic search (e.g., grid search, Bayesian optimization) to find window configurations that maximize WVQS.

---

## Environment & Dependencies

This repository assumes the following software environment:

- **Blender**: 4.3 (or later) – for 3D scene authoring and rendering
- **Python**: (e.g., 3.9 / 3.10 – adjust to your actual setup)
- **PyTorch**: for deep learning
- **Detectron2**: for Mask2Former integration
- **Mask2Former**: semantic segmentation model for window-view parsing

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
