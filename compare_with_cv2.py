import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule
from pycuda.driver import Stream
# Python helper functions
from PIL import Image
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as drv
import os
import time

# CUDA kernels
cuda_code = """
#define MY_PI 3.141592f
#define STRONG_EDGE 255
#define WEAK_EDGE 128
#define NO_EDGE 0

__global__ void gaussianBlur(float* kernel, unsigned char* source, unsigned char* target, int width, int height) {
    __shared__ unsigned char sharedMem[16 + 4][16 + 4];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedX = threadIdx.x + 2;
    int sharedY = threadIdx.y + 2;

    // 加载数据到共享内存
    sharedMem[sharedY][sharedX] = (x < width && y < height) ? source[y * width + x] : 0;

    // 修改边界填充（镜像处理）
    if (threadIdx.x < 2) {
        sharedMem[sharedY][sharedX - 2] = (x >= 2) ? source[y * width + x - 2] : source[y * width];
    }
    if (threadIdx.x >= blockDim.x - 2) {
        sharedMem[sharedY][sharedX + 2] = (x + 2 < width) ? source[y * width + x + 2] : source[y * width + (width - 1)];
    }
    if (threadIdx.y < 2) {
        sharedMem[sharedY - 2][sharedX] = (y >= 2) ? source[(y - 2) * width + x] : source[x];
    }
    if (threadIdx.y >= blockDim.y - 2) {
        sharedMem[sharedY + 2][sharedX] = (y + 2 < height) ? source[(y + 2) * width + x] : source[(height - 1) * width + x];
    }
    __syncthreads();

    // 计算高斯模糊，使用double精度
    double weightedSum = 0.0;
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            weightedSum += sharedMem[sharedY + j][sharedX + i] * kernel[(i + 2) * 5 + (j + 2)];
        }
    }
    weightedSum = min(max(weightedSum, 0.0), 255.0);
    target[y * width + x] = (unsigned char)(weightedSum);
}

__global__ void grayscale(unsigned char* rgbData, unsigned char* grayData, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // RGB 数据按顺序存储
        unsigned char r = rgbData[idx];
        unsigned char g = rgbData[idx + 1];
        unsigned char b = rgbData[idx + 2];

        // 加权平均
        grayData[y * width + x] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

__global__ void intensityGradient(unsigned char* img, int width, int height, float* magnitudes, float* directions) {
    __shared__ unsigned char sharedMem[16 + 2][16 + 2]; // 每个线程块 16x16 + 边界

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedX = threadIdx.x + 1;
    int sharedY = threadIdx.y + 1;

    // 加载数据到共享内存
    if (x < width && y < height) {
        sharedMem[sharedY][sharedX] = img[y * width + x];

        if (threadIdx.x == 0 && x > 0)
            sharedMem[sharedY][sharedX - 1] = img[y * width + x - 1];
        if (threadIdx.x == blockDim.x - 1 && x < width - 1)
            sharedMem[sharedY][sharedX + 1] = img[y * width + x + 1];
        if (threadIdx.y == 0 && y > 0)
            sharedMem[sharedY - 1][sharedX] = img[(y - 1) * width + x];
        if (threadIdx.y == blockDim.y - 1 && y < height - 1)
            sharedMem[sharedY + 1][sharedX] = img[(y + 1) * width + x];
    }
    __syncthreads(); // 确保所有线程加载完成

    if (x < width - 1 && y < height - 1 && x > 0 && y > 0) {
        int Gx = sharedMem[sharedY - 1][sharedX + 1] + 2 * sharedMem[sharedY][sharedX + 1] + sharedMem[sharedY + 1][sharedX + 1]
               - sharedMem[sharedY - 1][sharedX - 1] - 2 * sharedMem[sharedY][sharedX - 1] - sharedMem[sharedY + 1][sharedX - 1];
        int Gy = sharedMem[sharedY + 1][sharedX - 1] + 2 * sharedMem[sharedY + 1][sharedX] + sharedMem[sharedY + 1][sharedX + 1]
               - sharedMem[sharedY - 1][sharedX - 1] - 2 * sharedMem[sharedY - 1][sharedX] - sharedMem[sharedY - 1][sharedX + 1];

        magnitudes[y * width + x] = sqrtf((float)Gx * Gx + Gy * Gy);
        float direction = atan2f((float)Gy, (float)Gx) * 180.0f / MY_PI;
        directions[y * width + x] = (direction < 0) ? direction + 180.0f : direction;
    }
}

__global__ void hysteresis(unsigned char* img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    if (img[idx] != WEAK_EDGE) return;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int neighborIdx = (y + j) * width + (x + i);
            if (neighborIdx >= 0 && neighborIdx < width * height && img[neighborIdx] == STRONG_EDGE) {
                img[idx] = STRONG_EDGE;
                return;
            }
        }
    }
    img[idx] = NO_EDGE;
}
"""

# Compile CUDA code
mod = SourceModule(cuda_code)

# Retrieve kernels
gaussian_blur = mod.get_function("gaussianBlur")
grayscale = mod.get_function("grayscale")
hysteresis = mod.get_function("hysteresis")
intensity_gradient = mod.get_function("intensityGradient")


def allocate_pinned_memory(host_data):
    """
    Allocate pinned memory for host data.
    """
    pinned_memory = drv.pagelocked_empty_like(host_data)
    np.copyto(pinned_memory, host_data)  # Copy data into pinned memory
    return pinned_memory


def run_gaussian_blur(gray_data, kernel, width, height, stream):
    gray_data_gpu = drv.mem_alloc(gray_data.nbytes)
    kernel_gpu = drv.mem_alloc(kernel.nbytes)
    blurred_gpu = drv.mem_alloc(gray_data.nbytes)

    # Asynchronous data transfer
    drv.memcpy_htod_async(gray_data_gpu, gray_data, stream)
    drv.memcpy_htod_async(kernel_gpu, kernel, stream)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    gaussian_blur(kernel_gpu, gray_data_gpu, blurred_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size, stream=stream)

    blurred_data = np.empty_like(gray_data)
    drv.memcpy_dtoh_async(blurred_data, blurred_gpu, stream)
    stream.synchronize()
    return blurred_data


def run_intensity_gradient(gray_data, width, height, stream):
    gray_data_gpu = drv.mem_alloc(gray_data.nbytes)
    magnitudes_gpu = drv.mem_alloc(gray_data.nbytes * 4)
    directions_gpu = drv.mem_alloc(gray_data.nbytes * 4)

    drv.memcpy_htod_async(gray_data_gpu, gray_data, stream)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    intensity_gradient(gray_data_gpu, np.int32(width), np.int32(height), magnitudes_gpu, directions_gpu, block=block_size, grid=grid_size, stream=stream)

    magnitudes = np.empty((height, width), dtype=np.float32)
    directions = np.empty((height, width), dtype=np.float32)
    drv.memcpy_dtoh_async(magnitudes, magnitudes_gpu, stream)
    drv.memcpy_dtoh_async(directions, directions_gpu, stream)
    stream.synchronize()
    return magnitudes, directions


def run_hysteresis(magnitude_data, width, height, stream):
    magnitude_gpu = drv.mem_alloc(magnitude_data.nbytes)

    drv.memcpy_htod_async(magnitude_gpu, magnitude_data, stream)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    hysteresis(magnitude_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size, stream=stream)

    final_edges = np.empty_like(magnitude_data)
    drv.memcpy_dtoh_async(final_edges, magnitude_gpu, stream)
    stream.synchronize()
    return final_edges


def run_grayscale(rgb_data, width, height, stream):
    rgb_data_gpu = drv.mem_alloc(rgb_data.nbytes)
    gray_data_gpu = drv.mem_alloc(width * height)
    drv.memcpy_htod_async(rgb_data_gpu, rgb_data, stream)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    grayscale(rgb_data_gpu, gray_data_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size, stream=stream)

    gray_data = np.empty((height, width), dtype=np.uint8)
    drv.memcpy_dtoh_async(gray_data, gray_data_gpu, stream)
    stream.synchronize()
    return gray_data

# Timing Functions
def cuda_pipeline(image_rgb, kernel, stream):
    """
    Run CUDA pipeline: Grayscale -> Gaussian Blur -> Gradient -> Hysteresis.
    """
    height, width, _ = image_rgb.shape
    gray_cuda = run_grayscale(image_rgb, width, height, stream)
    blurred_cuda = run_gaussian_blur(gray_cuda, kernel, width, height, stream)
    magnitudes_cuda, _ = run_intensity_gradient(blurred_cuda, width, height, stream)
    edges_cuda = run_hysteresis((magnitudes_cuda > 100).astype(np.uint8) * 255, width, height, stream)
    return edges_cuda

# def opencv_pipeline(image):
#     """
#     Run OpenCV pipeline: Grayscale -> Gaussian Blur -> Sobel Gradient -> Canny Edges.
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
#     gradient = cv2.magnitude(cv2.Sobel(blurred, cv2.CV_64F, 1, 0), cv2.Sobel(blurred, cv2.CV_64F, 0, 1))
#     edges = cv2.Canny(blurred, 100, 200)
#     return edges

# Benchmark Execution Time
def benchmark_methods(image_path, kernel, stream):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # CUDA Method
    cuda_times = []
    for _ in range(5):
        start = time.time()
        cuda_pipeline(image_rgb, kernel, stream)
        cuda_times.append(time.time() - start)

    # OpenCV Method
    cv_times = []
    for _ in range(5):
        start = time.time()
        opencv_pipeline(image_rgb)
        cv_times.append(time.time() - start)

    return min(cuda_times), max(cuda_times), np.mean(cuda_times), min(cv_times), max(cv_times), np.mean(cv_times)

# Plot Runtime Comparisons
def plot_results(results):
    resolutions = [r['resolution'] for r in results]
    cuda_means = [r['cuda_mean'] for r in results]
    cv_means = [r['cv_mean'] for r in results]

    # Shading for min-max times
    cuda_min = [r['cuda_min'] for r in results]
    cuda_max = [r['cuda_max'] for r in results]
    cv_min = [r['cv_min'] for r in results]
    cv_max = [r['cv_max'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(resolutions, cuda_means, label="CUDA Mean Time", linewidth=2)
    plt.fill_between(resolutions, cuda_min, cuda_max, color="blue", alpha=0.2)

    plt.plot(resolutions, cv_means, label="OpenCV Mean Time", linewidth=2)
    plt.fill_between(resolutions, cv_min, cv_max, color="orange", alpha=0.2)

    plt.xlabel("Resolution")
    plt.ylabel("Execution Time (seconds)")
    plt.title("CUDA vs OpenCV Execution Time Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("runtime_comparison.png")
    plt.show()



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

def compare_methods(image_path, output_dir, resolution):
    """
    Compare CUDA-based implementation with OpenCV native methods.
    """
    # Load image using PIL and OpenCV
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape

    # Benchmark CUDA Implementation
    stream = drv.Stream()
    start_cuda = time.time()
    gray_cuda = run_grayscale(image_rgb, width, height, stream)
    blurred_cuda = run_gaussian_blur(gray_cuda, kernel, width, height, stream)
    magnitudes_cuda, _ = run_intensity_gradient(blurred_cuda, width, height, stream)
    edges_cuda = run_hysteresis((magnitudes_cuda > 100).astype(np.uint8) * 255, width, height, stream)
    end_cuda = time.time()

    # Benchmark OpenCV Implementation
    start_cv = time.time()
    gray_cv, blurred_cv, gradient_cv, edges_cv = opencv_pipeline(image)
    end_cv = time.time()

    # Save results
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Resolution: {resolution}", fontsize=14)

    # CUDA Results
    plt.subplot(2, 4, 1); plt.title("CUDA - Grayscale"); plt.imshow(gray_cuda, cmap='gray'); plt.axis('off')
    plt.subplot(2, 4, 2); plt.title("CUDA - Gaussian Blur"); plt.imshow(blurred_cuda, cmap='gray'); plt.axis('off')
    plt.subplot(2, 4, 3); plt.title("CUDA - Gradient Magnitude"); plt.imshow(magnitudes_cuda, cmap='gray'); plt.axis('off')
    plt.subplot(2, 4, 4); plt.title("CUDA - Edges"); plt.imshow(edges_cuda, cmap='gray'); plt.axis('off')

    # OpenCV Results
    plt.subplot(2, 4, 5); plt.title("OpenCV - Grayscale"); plt.imshow(gray_cv, cmap='gray'); plt.axis('off')
    plt.subplot(2, 4, 6); plt.title("OpenCV - Gaussian Blur"); plt.imshow(blurred_cv, cmap='gray'); plt.axis('off')
    plt.subplot(2, 4, 7); plt.title("OpenCV - Gradient Magnitude"); plt.imshow(gradient_cv, cmap='gray'); plt.axis('off')
    plt.subplot(2, 4, 8); plt.title("OpenCV - Edges"); plt.imshow(edges_cv, cmap='gray'); plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_{resolution}.png"))
    plt.close()

    # Print execution times
    print(f"Resolution: {resolution}x{resolution}")
    print(f"CUDA Execution Time: {end_cuda - start_cuda:.4f} seconds")
    print(f"OpenCV Execution Time: {end_cv - start_cv:.4f} seconds\n")

if __name__ == "__main__":
    dataset_path = "./cropped_datasets"
    kernel = np.array([
    1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256,
    4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
    6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256,
    4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
    1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256
    ], dtype=np.float32).reshape(5, 5)  # Gaussian kernel as before
    stream = Stream()

    results = []
    for resolution in ["128x128", "256x256", "320x320", "512x512", "640x640", "1024x1024"]:
        img_dir = os.path.join(dataset_path, resolution)
        img_file = os.listdir(img_dir)[0]  # Pick one image for simplicity
        img_path = os.path.join(img_dir, img_file)
        
        cuda_min, cuda_max, cuda_mean, cv_min, cv_max, cv_mean = benchmark_methods(img_path, kernel, stream)
        results.append({
            'resolution': resolution,
            'cuda_min': cuda_min, 'cuda_max': cuda_max, 'cuda_mean': cuda_mean,
            'cv_min': cv_min, 'cv_max': cv_max, 'cv_mean': cv_mean
        })
        print(f"Resolution {resolution} -> CUDA: {cuda_mean:.4f}s, OpenCV: {cv_mean:.4f}s")

    plot_results(results)
    
    
    
    # # Paths
    # dataset_path = "./cropped_datasets"
    # output_dir = "./comparison_results"
    # os.makedirs(output_dir, exist_ok=True)

    # # Gaussian Kernel
    # # kernel = np.array([
    # #     1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256,
    # #     4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
    # #     6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256,
    # #     4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
    # #     1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256
    # # ], dtype=np.float32)
    # kernel = np.array([
    # 1, 4, 6, 4, 1,
    # 4, 16, 24, 16, 4,
    # 6, 24, 36, 24, 6,
    # 4, 16, 24, 16, 4,
    # 1, 4, 6, 4, 1
    # ], dtype=np.float32)
    # kernel /= kernel.sum()
    # # Process each resolution folder
    # for resolution in ["256x256", "320x320", "512x512", "640x640", "1024x1024"]:
    #     img_dir = os.path.join(dataset_path, resolution)
    #     for img_file in os.listdir(img_dir):
    #         img_path = os.path.join(img_dir, img_file)
    #         compare_methods(img_path, output_dir, resolution)

    # print("Comparison completed. Check the output directory for results.")