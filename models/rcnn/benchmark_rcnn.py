import torch
import time
from rcnn.rcnn_train import get_model

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "runs/detect/rcnn_runs/production/FINAL_PRODUCTION_MODEL.pt"
    
    print("Loading model for benchmarking...")
    model = get_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Create dummy input (1 image, 3 channels, 640x640)
    dummy_input = torch.randn(1, 3, 640, 640).to(device)

    print("Warming up GPU kernels...")
    with torch.no_grad():
        for _ in range(20):
            _ = model([dummy_input]) # RCNN expects a list of image tensors

    print("Running benchmark (100 iterations)...")
    num_iterations = 100
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model([dummy_input])
            torch.cuda.synchronize() 

    total_time = time.time() - start_time
    avg_latency = (total_time / num_iterations) * 1000
    fps = 1000 / avg_latency

    print("\n--- INFERENCE BENCHMARK ---")
    print(f"Average Latency : {avg_latency:.2f} ms")
    print(f"Throughput      : {fps:.2f} FPS")

if __name__ == "__main__":
    run_benchmark()