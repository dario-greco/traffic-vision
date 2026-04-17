# BDD10k Dataset

**Git LFS Notice:** The dataset images in this repository are tracked using Git Large File Storage (LFS) rather than standard Git objects. 

## Overview
This directory contains the 10,000-image working subset sampled from the official [BDD100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/) dataset. It is specifically curated for detecting `traffic light` and `stop sign` classes to support the project's urban safety analytics objective.

## Data Splits
The 10,000 images are partitioned in the `images` folder:
* **Training (`train/`):** 7,000 images 
* **Validation (`val/`):** 1,000 images 
* **Testing (`test/`):** 2,000 images

## Label Formatting
All ground-truth annotations in the `labels/` directory have been standardized to the **YOLO format** (normalized `.txt` files containing: `class_id x_center y_center width height`). 

## Setup Instructions 

**Ensure Git LFS is Installed:** Before cloning or pulling, make sure Git LFS is initialized on your local machine or your cluster environment:
   ```bash
   git lfs install
