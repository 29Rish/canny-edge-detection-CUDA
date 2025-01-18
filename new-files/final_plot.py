import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pycuda_detection_new
import warp_optimaize_final
import memory_lock
import pycuda_shared
import cv2

def opencv_pipeline(image):
    """
    Apply OpenCV's native grayscale, Gaussian blur, and edge detection methods.
    """
    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Gradient Magnitude using Sobel
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 100, 200)
    
    return gray, blurred, gradient_magnitude, edges

def benchmark_methods(image_path):
    """
    Benchmark execution time of all CUDA edge detection methods and OpenCV.
    """
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # CUDA methods: Call directly and record returned runtime
    methods = {
        "Naive": pycuda_detection_new.edge_detection_pipeline,
        "Warp Optimized": warp_optimaize_final.edge_detection_pipeline,
        "Memory Lock": memory_lock.edge_detection_pipeline,
        "Shared Memory": pycuda_shared.edge_detection_pipeline
    }

    times = {}
    for method_name, method in methods.items():
        _, times[method_name] = method(image_path)  # Assume method returns runtime directly

    # OpenCV method
    start_cv = time.time()
    gray_cv, blurred_cv, gradient_cv, edges_cv = opencv_pipeline(image)
    end_cv = time.time()
    opencv_time = (end_cv - start_cv) 
    times["OpenCV"] = opencv_time

    return times

def plot_results(results):
    """
    Plot execution time comparisons across resolutions with average times and min-max range.
    """
    resolutions = [r['resolution'] for r in results]
    methods = ["Naive", "Warp Optimized", "Memory Lock", "Shared Memory", "OpenCV"]

    plt.figure(figsize=(10, 6))

    for method in methods:
        means = [r['mean'][method] for r in results]
        mins = [r['min'][method] for r in results]
        maxs = [r['max'][method] for r in results]

        # Plot mean execution time with min-max shaded region
        plt.plot(resolutions, means, label=method, marker='o', linewidth=2)
        plt.fill_between(resolutions, mins, maxs, alpha=0.2)

    plt.xlabel("Image Resolution")
    plt.ylabel("Execution Time (seconds)")
    plt.title("CUDA Methods vs OpenCV Execution Time Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure with high resolution
    plt.savefig("cuda_opencv_comparison_avg.png", dpi=600, bbox_inches="tight")

plt.show()

def main():
    dataset_path = "./cropped_datasets"
    resolutions = ["128x128", "256x256", "320x320", "512x512", "640x640", "1024x1024"]

    results = []
    for resolution in resolutions:
        print(f"Processing resolution: {resolution}")
        img_dir = os.path.join(dataset_path, resolution)
        img_files = os.listdir(img_dir)

        # Store all execution times for current resolution
        method_times = {"Naive": [], "Warp Optimized": [], "Memory Lock": [], "Shared Memory": [], "OpenCV": []}

        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            times = benchmark_methods(img_path)  # Benchmark for a single image
            for method, time_value in times.items():
                method_times[method].append(time_value)

        # Calculate average, min, and max for each method
        resolution_stats = {'resolution': resolution, 'mean': {}, 'min': {}, 'max': {}}
        for method, time_list in method_times.items():
            resolution_stats['mean'][method] = np.mean(time_list)
            resolution_stats['min'][method] = np.min(time_list)
            resolution_stats['max'][method] = np.max(time_list)

        results.append(resolution_stats)
        print(f"Results for {resolution}: {resolution_stats}")

    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main()