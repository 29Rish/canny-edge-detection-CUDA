# Canny Edge Detection Using PyCUDA

## Project Description
This project implements the Canny edge detection algorithm using PyCUDA, leveraging GPU acceleration for enhanced performance. The project is based on the research paper *"Canny Edge Detection on NVIDIA CUDA"* by Yuancheng “Mike” Luo and Ramani Duraiswami. By exploiting the massively parallel architecture of GPUs, this implementation achieves substantial speedups compared to traditional CPU-based methods.

## Introduction to Canny Edge Detection

The **Canny edge detector** is a widely used edge detection algorithm in computer vision. It is recognized for its robustness and ability to accurately detect edges while minimizing false positives. The algorithm operates in multiple steps:

1. **Gaussian Smoothing**: The image is convolved with a Gaussian filter to reduce noise and spurious details.
2. **Gradient Calculation**: Sobel operators are used to compute the gradients in the x and y directions. The gradient magnitude and direction are derived from these computations.
3. **Non-Maximum Suppression**: Suppresses all non-maximum gradient values in the gradient direction to thin out edges.
4. **Hysteresis Thresholding**: Applies two thresholds (high and low) to classify edges into strong, weak, and non-edges. Weak edges are retained if they are connected to strong edges.

### GPU Acceleration with CUDA
The GPU implementation of the Canny edge detector utilizes the CUDA framework for significant performance gains. CUDA allows parallel execution of image processing tasks across thousands of GPU cores. This project leverages:
- **Shared Memory**: To reduce global memory access latency.
- **Memory Coalescing**: To optimize memory access patterns.
- **Thread Blocks**: Each thread processes a pixel, enabling highly parallel operations.

Here’s the README content formatted in Markdown syntax:  


## How to Run

1. **Run Naive Algorithm:**  
   To perform edge detection using the naive algorithm, execute the following script:  
   ```bash
   python pycuda_detection_new(1).py
   ```

2. **Run All Detection Methods:**  
   To run all edge detection methods together, execute:  
   ```bash
   python pycuda_combined_new.py
   ```

## Requirements

Ensure the following dependencies are installed on your system:

- **CUDA Toolkit**: Required for GPU-accelerated computation.  
- **Keras**: For machine learning-related processing.  
- **TensorFlow**: Backend for Keras and additional processing.
- **NCU command-line tool**
- **Python**
- **Seaborn and Matplotlib libraries**
 

## Notes

- Make sure your environment is properly configured to support CUDA.  
- Use Python 3.7+ for compatibility with the scripts.


## Scripts

### `ncu_create.sh`
This script is used to generate NCU (NVIDIA Compute Utility) performance reports. Running this script will profile a CUDA application and output a report that can be analyzed to understand the performance characteristics of the application.

#### Usage
To run the `ncu_create.sh` script:
```bash
bash ncu_create.sh
```

### `compare_plot.sh`
This script generates comparative images of NCU reports. It compares different performance runs to visually highlight the performance differences.

#### Usage
To run the `compare_plot.sh` script:
```bash
bash compare_plot.sh
```


## Algorithm Stages

### 1. Gaussian Smoothing
- **Purpose**: Reduces image noise while preserving edge details.
- **Implementation**: A separable Gaussian filter is applied in two passes—horizontal and vertical—to reduce computational complexity from \(O(M^2)\) to \(O(2M)\) operations per pixel.

### 2. Gradient Computation
- **Purpose**: Detects intensity changes in the image.
- **Implementation**: Sobel filters compute the gradients (G_x and G_y) in the x and y directions. The gradient magnitude and direction are calculated as:

  **Gradient Magnitude:**  
  G = sqrt(G_x² + G_y²)

  **Gradient Direction:**  
  θ = arctan(G_y / G_x)

- **Quantization**: The gradient direction is quantized to one of four main directions (0°, 45°, 90°, and 135°).

### 3. Non-Maximum Suppression
- **Purpose**: Eliminates pixels that are not part of the local maxima in the gradient direction, producing thin edges.
- **Implementation**: Each pixel’s gradient magnitude is compared with the magnitudes of its neighbors in the gradient direction. Non-maximum values are suppressed.

### 4. Hysteresis Thresholding and Connected Components
- **Purpose**: Retains edges classified as strong and connects weak edges that are part of strong edge chains.
- **Implementation**:  
  - Two thresholds (T_low and T_high) are applied to classify pixels into:
    - **Strong edges**: Gradient magnitude > T_high
    - **Weak edges**: T_low ≤ Gradient magnitude ≤ T_high
    - **Non-edges**: Gradient magnitude < T_low
  - A breadth-first search (BFS) identifies weak edges connected to strong edges.


## Advantages of GPU Implementation
- **Parallel Processing**: Tasks such as convolution and non-maximum suppression are performed independently for each pixel.
- **Memory Optimization**: Efficient use of shared memory and coalesced access patterns minimizes the latency of global memory operations.
- **Significant Speedups**: Benchmarks show performance improvements of up to 80x compared to MATLAB implementations and 3-5x compared to optimized CPU libraries like OpenCV.

## Limitations
- Requires an NVIDIA GPU with CUDA support.
- Performance is influenced by the complexity of input images and GPU architecture.

## Results
- The implementation is tested on standard images (e.g., Lena, Mandrill).
- Benchmarks show linear scalability with increasing image resolution.
- Significant speedup compared to both MATLAB and OpenCV implementations.

## References
1. Luo, Y., & Duraiswami, R. *Canny Edge Detection on NVIDIA CUDA*. [Research Paper]((https://ieeexplore.ieee.org/document/4563088))
2. NVIDIA CUDA Programming Guide: [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)
3. J.F. Canny. *A Computational Approach to Edge Detection*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1986.

---

This README provides a comprehensive overview of the theoretical and practical aspects of implementing the Canny edge detector using PyCUDA. For details on running the code and installation instructions, refer to the **Usage** and **Installation** sections.
