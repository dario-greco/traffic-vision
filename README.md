# Traffic Vision: Traffic Sign & Light Detection

## Overview
This project builds and evaluates a deep learning pipeline for detecting **traffic signs** and **traffic lights** in urban dashcam footage. The goal is to support safety analytics use cases in ride-sharing platforms by identifying critical traffic control devices in a driver’s field of view.

Rather than directly assessing driver behavior, this system provides a **foundational perception layer** that can enable downstream applications such as:
- Automated trip auditing  
- Real-time driver alerts  
- Post-incident analysis  

---

## 📊 Dataset
We use a **10,000-image subset (BDDK10k)** sampled from the [BDD100K dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd/), which contains real-world dashcam imagery from urban environments including:
- New York  
- Berkeley  
- San Francisco  
- Greater Bay Area  

### Why BDD100K?
- **Scale & Diversity:** Large, varied urban driving scenarios  
- **Domain Relevance:** Real dashcam footage (not generic images)  
- **No Data Leakage:** Avoids overlap with common pretrained datasets like COCO  
- **Better Generalization:** More realistic deployment conditions  

---

## Objective
Detect traffic control devices with strong performance across:
- **Accuracy** (mAP, precision, recall)  
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

##  Methodology

### Data Processing
- Standardized annotation formats across models  
- Unified label schema (traffic signs, traffic lights)  
- Train / validation / test split with no overlap  
- Consistent augmentation pipeline across models  

### Training
Each model was trained on the same dataset subset under comparable conditions to ensure fairness.

### Evaluation Metrics
- **Primary:** mAP, Precision, Recall  
- **Secondary:** Inference Time  

### Model Selection Criteria
We selected the best model based on the **accuracy–efficiency trade-off**, prioritizing strong detection performance while considering deployment feasibility.

---

## Results

- **Best Model:** Faster R-CNN  
- **Final Step:** Hyperparameter tuning on RCNN  
- **Outcome:** Fine-tuned RCNN achieved the strongest overall performance  

> RCNN demonstrated superior localization and detection reliability compared to other models under our evaluation framework.
---

## Project Structure
traffic-vision/
│── data/ # Dataset and splits
│── models/ # Model definitions and configs
│── training/ # Training scripts
│── evaluation/ # Metrics and evaluation scripts
│── results/ # Outputs, plots, checkpoints
│── notebooks/ # Experimentation and visualization
│── README.md


## How to Run

### Clone the repo
```bash
git clone https://github.com/dario-greco/traffic-vision.git
cd traffic-vision
```
### Install Dependencies 
```bash
pip install -r requirements.txt
```
### Prepare Dataset
- Download BDD100K
- Sample or use provided BDDK10k subset
- Format annotations as required per model
### Train Model
```bash
python train.py --model rcnn
```
### Evaluate Performance 
```bash
python evaluate.py --model rcnn
```
## Conclusion 
This project establishes a full pipeline for traffic control device detection in dashcam footage. Among the evaluated models, Faster D-FINE emerges as the strongest candidate, particularly after hyperparameter tuning, making it a solid baseline for future deployment-oriented safety systems.
