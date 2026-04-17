# Traffic Vision: Traffic Sign & Light Detection

## Overview

This project builds and evaluates a deep learning perception pipeline designed to reliably detect **traffic signs** and **traffic lights** in urban dashcam footage. Developed witha  ride-sharing safety analytics use case in mind, these perception layers are the first step toward enabling downstream safety applications such as automated trip auditing, post-incident analysis, and real-time driving context awareness.

---
## Dataset: BDD10k
To ensure the models generalize to real-world driving conditions, we utilized a **10,000-image subset (BDDK10k)** randomly sampled from the [BDD100K Dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd/). 

* **Domain Relevance:** Imagery is captured entirely from a vehicle’s dashboard perspective across New York, Berkeley, San Francisco, and the Greater Bay Area.
* **Native Diversity:** The dataset features an approximately even split between daytime and nighttime driving, with significant meteorological variability (rain, snow, fog, overcast). This native diversity largely bypasses the need for synthetic data augmentation.
* **Data Split:** 7,000 Training | 1,000 Validation | 2,000 Held-out Test.
---

## Objective
Detect traffic control devices with strong performance across:
- **Accuracy** (mAP)  
- **Efficiency** (inference speed)  
---

## Models Evaluated
We trained and evaluated four distinct object detection architectures to find the optimal balance between detection accuracy and inference efficiency:

| Model | Architecture Type | Key Characteristic |
| :--- | :--- | :--- |
| **YOLOv8n** | Single-stage CNN | Extremely fast inference; acts as our baseline. |
| **Swin Transformer** | Transformer-based | Hierarchical feature extraction with shifted-window attention. |
| **Faster R-CNN** | Two-stage Detector | High-quality region proposals for strong localization. |
| **D-FINE** | DETR-based (Modern) | Uses Fine-grained Distribution Refinement (FDR) without NMS. |

Models were evaluated on a held-out test set using Mean Average Precision (mAP) across varying IoU thresholds and Inference Speed (images per second)

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
This project uses uv for fast dependency management (as specified in `pyproject.toml`):

```bash
uv sync
uv pip install -r models/D-FINE/requirements.txt
```

### Evaluation

To run the automated comparison script across all trained models and generate the final metrics in the results/ folder:

```bash
uv run python scripts/evaluation.py
```


## References
- [BDD100K Dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd/)
