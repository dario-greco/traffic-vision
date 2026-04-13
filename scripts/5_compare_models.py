"""
Evaluation and Comparison Script for YOLO vs Faster R-CNN
Compares detection performance, inference speed, and accuracy metrics between both models.
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Import YOLO
from ultralytics import YOLO

# Import Faster R-CNN components
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "yolo_dataset")
YOLO_IMAGES_DIR = os.path.join(DATASET_DIR, "images")
YOLO_LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# Output directory
EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "detect", "comparison")
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)


def load_yolo_model(model_path):
    """Load a trained YOLO model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None


def load_rcnn_model(model_weights_path, num_classes=3):
    """Load a trained Faster R-CNN model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        print(f"Error loading Faster R-CNN model: {e}")
        return None, None


def evaluate_yolo(model, test_img_dir, test_label_dir, confidence_threshold=0.5):
    """Evaluate YOLO model on test set"""
    print("\nEvaluating YOLO model...")
    
    yolo_detections = {}
    inference_times = []
    
    img_files = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))]
    
    for img_file in tqdm(img_files, desc="YOLO Inference"):
        img_path = os.path.join(test_img_dir, img_file)
        
        # Inference
        import time
        start_time = time.time()
        results = model.predict(img_path, conf=confidence_threshold, verbose=False)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Parse results
        detections = []
        for r in results:
            if r.boxes is not None:
                for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    detections.append({
                        "bbox": box.cpu().numpy().tolist(),
                        "confidence": float(conf),
                        "class": int(cls)
                    })
        
        yolo_detections[img_file] = detections
    
    metrics = {
        "model": "YOLOv8",
        "num_images": len(img_files),
        "avg_inference_time": np.mean(inference_times),
        "std_inference_time": np.std(inference_times),
        "min_inference_time": np.min(inference_times),
        "max_inference_time": np.max(inference_times),
        "total_detections": sum(len(d) for d in yolo_detections.values()),
    }
    
    return yolo_detections, metrics


def evaluate_rcnn(model, device, test_img_dir, test_label_dir, confidence_threshold=0.5):
    """Evaluate Faster R-CNN model on test set"""
    print("\nEvaluating Faster R-CNN model...")
    
    rcnn_detections = {}
    inference_times = []
    
    img_files = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))]
    
    for img_file in tqdm(img_files, desc="R-CNN Inference"):
        img_path = os.path.join(test_img_dir, img_file)
        
        # Load and preprocess image
        image = torchvision.io.read_image(img_path).float().to(device) / 255.0
        
        # Inference
        import time
        start_time = time.time()
        with torch.no_grad():
            outputs = model([image])
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Parse results
        detections = []
        output = outputs[0]
        
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score >= confidence_threshold:
                detections.append({
                    "bbox": box.cpu().numpy().tolist(),
                    "confidence": float(score),
                    "class": int(label) - 1  # Convert back to 0-indexed
                })
        
        rcnn_detections[img_file] = detections
    
    metrics = {
        "model": "Faster R-CNN ResNet50",
        "num_images": len(img_files),
        "avg_inference_time": np.mean(inference_times),
        "std_inference_time": np.std(inference_times),
        "min_inference_time": np.min(inference_times),
        "max_inference_time": np.max(inference_times),
        "total_detections": sum(len(d) for d in rcnn_detections.values()),
    }
    
    return rcnn_detections, metrics


def compare_models():
    """Main comparison function"""
    print("=" * 60)
    print("YOLO vs Faster R-CNN Comparison")
    print("=" * 60)
    
    # Paths
    yolo_model_path = os.path.join(BASE_DIR, "runs", "detect", "traffic_v8n_runs", "traffic_model", "weights", "best.pt")
    rcnn_model_path = os.path.join(BASE_DIR, "runs", "detect", "rcnn_runs", "traffic_model_rcnn", "faster_rcnn_best.pt")
    test_img_dir = os.path.join(YOLO_IMAGES_DIR, "val")
    test_label_dir = os.path.join(YOLO_LABELS_DIR, "val")
    
    if not os.path.exists(test_img_dir):
        print(f"Test image directory not found: {test_img_dir}")
        return
    
    results_comparison = {}
    
    # Evaluate YOLO if available
    if os.path.exists(yolo_model_path):
        yolo_model = load_yolo_model(yolo_model_path)
        if yolo_model:
            yolo_detections, yolo_metrics = evaluate_yolo(yolo_model, test_img_dir, test_label_dir)
            results_comparison["yolo"] = yolo_metrics
            print("\nYOLO Results:")
            print(json.dumps(yolo_metrics, indent=2))
    else:
        print(f"YOLO model not found at {yolo_model_path}")
    
    # Evaluate Faster R-CNN if available
    if os.path.exists(rcnn_model_path):
        rcnn_model, device = load_rcnn_model(rcnn_model_path)
        if rcnn_model:
            rcnn_detections, rcnn_metrics = evaluate_rcnn(rcnn_model, device, test_img_dir, test_label_dir)
            results_comparison["rcnn"] = rcnn_metrics
            print("\nFaster R-CNN Results:")
            print(json.dumps(rcnn_metrics, indent=2))
    else:
        print(f"Faster R-CNN model not found at {rcnn_model_path}")
    
    # Save comparison results
    comparison_path = os.path.join(EVAL_OUTPUT_DIR, "model_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(results_comparison, f, indent=2)
    print(f"\nComparison results saved to {comparison_path}")
    
    # Create comparison plots
    if "yolo" in results_comparison and "rcnn" in results_comparison:
        create_comparison_plots(results_comparison)
    
    return results_comparison


def create_comparison_plots(results):
    """Create comparison plots for YOLO vs R-CNN"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = ["YOLO", "Faster R-CNN"]
    yolo_metrics = results.get("yolo", {})
    rcnn_metrics = results.get("rcnn", {})
    
    # Inference time comparison
    if yolo_metrics and rcnn_metrics:
        ax = axes[0, 0]
        inference_times = [
            yolo_metrics.get("avg_inference_time", 0),
            rcnn_metrics.get("avg_inference_time", 0)
        ]
        colors = ['#FF6B6B', '#4ECDC4']
        ax.bar(models, inference_times, color=colors, alpha=0.7)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Average Inference Time per Image')
        ax.set_ylim(0, max(inference_times) * 1.2)
        for i, v in enumerate(inference_times):
            ax.text(i, v + 0.001, f'{v:.4f}s', ha='center', va='bottom')
        
        # Total detections comparison
        ax = axes[0, 1]
        total_detections = [
            yolo_metrics.get("total_detections", 0),
            rcnn_metrics.get("total_detections", 0)
        ]
        ax.bar(models, total_detections, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Detections')
        ax.set_title('Total Detections on Test Set')
        for i, v in enumerate(total_detections):
            ax.text(i, v + 10, str(v), ha='center', va='bottom')
        
        # Inference time distribution
        ax = axes[1, 0]
        speed_metrics = ['Min Time', 'Avg Time', 'Max Time']
        yolo_times = [
            yolo_metrics.get("min_inference_time", 0),
            yolo_metrics.get("avg_inference_time", 0),
            yolo_metrics.get("max_inference_time", 0),
        ]
        rcnn_times = [
            rcnn_metrics.get("min_inference_time", 0),
            rcnn_metrics.get("avg_inference_time", 0),
            rcnn_metrics.get("max_inference_time", 0),
        ]
        x = np.arange(len(speed_metrics))
        width = 0.35
        ax.bar(x - width/2, yolo_times, width, label='YOLO', color='#FF6B6B', alpha=0.7)
        ax.bar(x + width/2, rcnn_times, width, label='Faster R-CNN', color='#4ECDC4', alpha=0.7)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Inference Time Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(speed_metrics)
        ax.legend()
        
        # Summary text
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
MODEL COMPARISON SUMMARY

YOLO v8:
  • Avg Inference: {yolo_metrics.get('avg_inference_time', 0):.4f}s
  • Total Detections: {yolo_metrics.get('total_detections', 0)}
  • Images Tested: {yolo_metrics.get('num_images', 0)}

Faster R-CNN:
  • Avg Inference: {rcnn_metrics.get('avg_inference_time', 0):.4f}s
  • Total Detections: {rcnn_metrics.get('total_detections', 0)}
  • Images Tested: {rcnn_metrics.get('num_images', 0)}

SpeedUp (YOLO/R-CNN):
  • {yolo_metrics.get('avg_inference_time', 1) / rcnn_metrics.get('avg_inference_time', 1):.2f}x faster
        """
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_path = os.path.join(EVAL_OUTPUT_DIR, "model_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    compare_models()
