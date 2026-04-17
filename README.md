# Traffic Vision: Traffic Sign & Light Detection

## Overview
This project builds and evaluates a deep learning pipeline for detecting **traffic signs** and **traffic lights** in urban dashcam footage. The goal is to support safety analytics use cases in ride-sharing platforms by identifying critical traffic control devices in a driver’s field of view.
---

## Dataset
We use a **10,000-image subset (BDDK10k)** sampled from the [BDD100K dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd/), which contains real-world dashcam imagery from urban environments including:
- New York  
- Berkeley  
- San Francisco  
- Greater Bay Area 

The data is avaliable as in the `data_final/` folder.

---
### Why BDD100K?
- **Scale & Diversity:** Large, varied urban driving scenarios and weather enviroments
- **Domain Relevance:** Real dashcam footage 
---

## Objective
Detect traffic control devices with strong performance across:
- **Accuracy** (mAP)  
- **Efficiency** (inference speed)  
---

## Models Evaluated
We train and compare four object detection architectures:

| Model | Type | Key Strength |
|------|------|-------------|
| **YOLO** | Single-stage | Fast inference (baseline) |
| **Swin Transformer** | Transformer-based | Strong contextual understanding |
| **Faster R-CNN (RCNN)** | Two-stage | High localization accuracy |
| **D-FINE** | Modern detector | Competitive performance benchmark |

---


## Project Structure
```
├── models/                          # Model implementations and weights
│   ├── D-FINE/                     # D-FINE model implementation
│   │   ├── train_dfine.py          # Training script
│   │   ├── eval_dfine.py           # Evaluation script
│   │   ├── utils.py                # Utility functions
│   │   ├── src/                    # Model source code
│   │
│   ├── rcnn/                       # Faster R-CNN implementation
│   │   ├── rcnn_train.py           # Training script
│   │   ├── rcnn_untuned.pt         # Trained weights
│   │   └── hpo/                    # Hyperparameter optimization
│   │       ├── rcnn_hpo.py         # HPO script
│   │
│   ├── yolo/                       # YOLOv8 implementation
│   │   ├── yolo_train.py           # Training script
│   │   └── yolo_untuned.pt         # Trained weights
│   │
│   └── swin_transformer/           # Swin Transformer implementation
│       ├── swinTransformer.py      # Model implementation
│       └── best_model.pt           # Best trained weights
│
├── data_final/                     # Dataset (generated locally)
│   ├── images/                     # Training/validation/test images
│   └── labels/                     # Corresponding labels
│
├── scripts/                        # Model performance comparison scripts
├── logs/                           # Training logs
├── results/                        # Output results and visualizations
└── runs/                           # YOLOv8 run artifacts
```


## How to Run Evaluation

### Clone the repo
```bash
git clone https://github.com/dario-greco/traffic-vision.git
cd traffic-vision
```
### Install Dependencies 
Using `uv` (as specified in `pyproject.toml`):

```bash
uv sync
uv pip install -r models/D-FINE/requirements.txt
```

### Evaluation

To compare all the trained models run:

```bash
uv run python scripts/evaluation.py
```

the output will be stored in the `results/` folder.

## References
- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
