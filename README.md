# Mammogram Image Classification – Data Loading & Evaluation

This repository provides a reusable PyTorch pipeline for preparing, augmenting, and evaluating mammogram images (or other medical image datasets) for deep learning tasks such as classification or quality assessment.

The core functionality is implemented across four scripts:

- `dataloader_final.py` – Standalone dataloader module for projects.
- `model_final_new.py` – Model-ready dataloader for training pipelines.
- `train.py` – Data preparation utilities intended for training experiments.
- `eval.py` – Data preparation utilities intended for evaluation/testing.

All four share a common dataset design with:

- CLAHE preprocessing for contrast enhancement.
- Configurable transforms & augmentations.
- Easy train/val/test split handling.

---

## Project Structure

your_project/
|
+-- dataloader_final.py
+-- model_final_new.py
+-- train.py
+-- eval.py
|
+-- README.md
+-- data/
+-- train/
| +-- class_1/
| | +-- image1.jpg
| | +-- ...
| +-- class_2/
| +-- ...
+-- val/
| +-- class_1/
| +-- class_2/
+-- test/
+-- class_1/
+-- class_2/


Inside each, create **one folder per class containing the images**.

---

## Key Features

- **CLAHE Contrast Enhancement**  
  Each image is first converted to grayscale and processed using CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve visibility of structures, then converted back to RGB.

- **Robust Data Augmentation**  
  Training transforms include:
  - Random horizontal flip
  - Small random rotation (±5°)
  - Random affine transformations (translation, scale)
  - Light brightness/contrast jitter

- **ImageNet Normalization**  
  Output tensors are normalized using ImageNet statistics:
  - `mean = [0.485, 0.456, 0.406]`
  - `std  = [0.229, 0.224, 0.225]`

- **Safe Loading**  
  Corrupted or unreadable images are skipped with a warning, replaced by a zero tensor.

---

## Installation

1. Clone the Repository

git clone https://github.com/meli-jegi/mammo_classification.git
cd mammo_classification

Create Environment (Recommended)

conda create -n mammogram python=3.10
conda activate mammogram

Install Dependencies
pip install torch torchvision pillow opencv-python numpy

Usage

Each script exposes a get_dataloaders() function:

```python
data_dir = "/path/to/data"
batch_size = 32
img_size = 224

dataloaders, dataset_sizes, class_names = get_dataloaders(
    data_dir=data_dir,
    batch_size=batch_size,
    img_size=img_size,
    num_workers=4
)
```
Returns
dataloaders: Dictionary with train, val, and test PyTorch DataLoader objects.

dataset_sizes: Dictionary of dataset sizes.

class_names: Sorted list of class labels.

File-by-File Details

dataloader_final.py

Purpose: Clean, modular dataloader for use in training or evaluation scripts.
Use Case: Import this in your custom training loop or model code.

model_final_new.py

Purpose: Identical to dataloader_final.py, intended to be imported directly inside model training scripts.
Use Case: For projects that require a dedicated model-centric dataloader.

train.py

Purpose: Prepares dataloaders specifically for training experiments.
Typical Flow:

Import get_dataloaders from this file.

Build your training loop (torch.nn, optimizer, loss, etc.).

Use dataloaders['train'] and dataloaders['val'].

eval.py

Purpose: Same as above but intended for evaluation or testing.
Typical Flow:

Load trained model.

Import get_dataloaders from eval.py.

Run predictions on dataloaders['test'].


Example Training Skeleton

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from train import get_dataloaders

# Get data
dataloaders, dataset_sizes, class_names = get_dataloaders(
    data_dir="data",
    batch_size=32,
    img_size=224
)
```

# Simple model
```python
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

# Training loop (simplified)
```python
for epoch in range(10):
    model.train()
    for inputs, labels, _ in dataloaders['train']:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

Customization
Image Size: Change img_size argument in get_dataloaders.

Augmentations: Modify train_transform in any script to add/remove transformations.

CLAHE Parameters: Adjust clipLimit or tileGridSize in cv2.createCLAHE to control contrast enhancement.

Dataset Splits: Ensure correct data placement (train, val, test) to avoid data leakage.

Example Evaluation Skeleton


# Load model
```python
from eval import get_dataloaders
import torch

model = torch.load("model.pth")
model.eval()

dataloaders, _, class_names = get_dataloaders(
    data_dir="data",
    batch_size=32,
    img_size=224
)

correct, total = 0, 0
with torch.no_grad():
    for inputs, labels, _ in dataloaders['test']:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total:.2%}")
```
