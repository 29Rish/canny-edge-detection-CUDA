import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule

# Python helper functions
from PIL import Image
import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
# CUDA kernels
cuda_code = """
#define MY_PI 3.141592f
#define STRONG_EDGE 255
#define WEAK_EDGE 128
#define NO_EDGE 0

__global__ void gaussianBlur(float* kernel, unsigned char* source, unsigned char* target, int width, int height) {
    __shared__ unsigned char sharedMem[16 + 4][16 + 4]; // 每个 block 分配共享内存

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedX = threadIdx.x + 2;
    int sharedY = threadIdx.y + 2;

    // 加载数据到共享内存 (突发访问)
    if (x < width && y < height) {
        sharedMem[sharedY][sharedX] = source[y * width + x];

        // 加载 halo 数据 (边界部分)
        if (threadIdx.x < 2) { // 左边界
            sharedMem[sharedY][sharedX - 2] = (x >= 2) ? source[y * width + x - 2] : 0;
        }
        if (threadIdx.x >= blockDim.x - 2) { // 右边界
            sharedMem[sharedY][sharedX + 2] = (x + 2 < width) ? source[y * width + x + 2] : 0;
        }
        if (threadIdx.y < 2) { // 上边界
            sharedMem[sharedY - 2][sharedX] = (y >= 2) ? source[(y - 2) * width + x] : 0;
        }
        if (threadIdx.y >= blockDim.y - 2) { // 下边界
            sharedMem[sharedY + 2][sharedX] = (y + 2 < height) ? source[(y + 2) * width + x] : 0;
        }
    }
    __syncthreads(); // 确保所有线程加载完成

    // 计算高斯模糊
    if (x < width && y < height) {
        float weightedSum = 0.0f;

        #pragma unroll 5
        for (int i = -2; i <= 2; i++) {
            #pragma unroll 5
            for (int j = -2; j <= 2; j++) {
                weightedSum += sharedMem[sharedY + j][sharedX + i] * kernel[(i + 2) * 5 + (j + 2)];
            }
        }

        target[y * width + x] = (unsigned char)weightedSum;
    }
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
    plt.suptitle(f"Resolution: {resolution}x{resolution}", fontsize=14)

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


def visualize_all_kernels(image_path):
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    height, width, _ = img_array.shape

    # Allocate pinned memory for host data
    pinned_img_array = allocate_pinned_memory(img_array)

    # Create CUDA stream
    stream = drv.Stream()

    # Grayscale conversion
    gray_img = run_grayscale(pinned_img_array, width, height, stream)

    # Gaussian Blur
    kernel = np.array([
        1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256,
        4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
        6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256,
        4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
        1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256
    ], dtype=np.float32)
    blurred_img = run_gaussian_blur(gray_img, kernel, width, height, stream)

    # Intensity Gradient
    magnitudes, directions = run_intensity_gradient(blurred_img, width, height, stream)

    # Hysteresis
    edges = run_hysteresis((magnitudes > 100).astype(np.uint8) * 255, width, height, stream)

    # Visualization
    plt.figure(figsize=(20, 15))

    plt.subplot(2, 3, 1)
    plt.title("Original RGB Image")
    plt.imshow(img_array)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Grayscale Image")
    plt.imshow(gray_img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Gaussian Blurred Image")
    plt.imshow(blurred_img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Gradient Magnitudes")
    plt.imshow(magnitudes, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Gradient Directions")
    plt.imshow(directions, cmap='hsv')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Final Edges after Hysteresis")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.savefig("output_warp_optimaize.png")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_all_kernels("test.png")