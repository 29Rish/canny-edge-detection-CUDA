import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule
# Python helper functions
from PIL import Image
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import time

# CUDA kernels
cuda_code = """
#define M_PI 3.141592f
#define STRONG_EDGE 255
#define WEAK_EDGE 128
#define NO_EDGE 0

__global__ void gaussianBlur_shared(float* kernel, unsigned char* source, unsigned char* target, int width, int height) {
    __shared__ unsigned char sharedMem[16 + 4][16 + 4]; // 16x16 threads + 2 padding on each side
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sharedX = threadIdx.x + 2; // Offset for padding
    int sharedY = threadIdx.y + 2;

    // Load data into shared memory
    if (x < width && y < height) {
        sharedMem[sharedY][sharedX] = source[y * width + x];

        // Load halo (boundary) data
        if (threadIdx.x < 2 && x >= 2)
            sharedMem[sharedY][sharedX - 2] = source[y * width + x - 2];
        if (threadIdx.x >= blockDim.x - 2 && x < width - 2)
            sharedMem[sharedY][sharedX + 2] = source[y * width + x + 2];
        if (threadIdx.y < 2 && y >= 2)
            sharedMem[sharedY - 2][sharedX] = source[(y - 2) * width + x];
        if (threadIdx.y >= blockDim.y - 2 && y < height - 2)
            sharedMem[sharedY + 2][sharedX] = source[(y + 2) * width + x];
    }
    __syncthreads();

    // Apply Gaussian blur
    if (x < width && y < height) {
        float weightedSum = 0.0f;
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                weightedSum += sharedMem[sharedY + j][sharedX + i] * kernel[(i + 2) * 5 + (j + 2)];
            }
        }
        target[y * width + x] = (unsigned char)weightedSum;
    }
}

__global__ void grayscale_shared(unsigned char* rgbData, unsigned char* grayData, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3; // RGB channels
    grayData[y * width + x] = (unsigned char)((0.299f * rgbData[idx]) + (0.587f * rgbData[idx + 1]) + (0.114f * rgbData[idx + 2]));
}

__global__ void intensityGradient_shared(unsigned char* img, int width, int height, float* magnitudes, float* directions) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    int idx = y * width + x;
    int Gx = img[(y - 1) * width + (x + 1)] + 2 * img[y * width + (x + 1)] + img[(y + 1) * width + (x + 1)]
           - img[(y - 1) * width + (x - 1)] - 2 * img[y * width + (x - 1)] - img[(y + 1) * width + (x - 1)];
    int Gy = img[(y + 1) * width + (x - 1)] + 2 * img[(y + 1) * width + x] + img[(y + 1) * width + (x + 1)]
           - img[(y - 1) * width + (x - 1)] - 2 * img[(y - 1) * width + x] - img[(y - 1) * width + (x + 1)];

    magnitudes[idx] = sqrtf(Gx * Gx + Gy * Gy);
    directions[idx] = atan2f(Gy, Gx) * 180.0f / M_PI;
    if (directions[idx] < 0) directions[idx] += 180.0f;
}

__global__ void hysteresis_shared(unsigned char* img, int width, int height) {
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
gaussian_blur = mod.get_function("gaussianBlur_shared")
grayscale = mod.get_function("grayscale_shared")
hysteresis = mod.get_function("hysteresis_shared")
intensity_gradient = mod.get_function("intensityGradient_shared")


def run_gaussian_blur(gray_data, kernel, width, height):
    gray_data_gpu = drv.mem_alloc(gray_data.nbytes)
    kernel_gpu = drv.mem_alloc(kernel.nbytes)
    blurred_gpu = drv.mem_alloc(gray_data.nbytes)

    drv.memcpy_htod(gray_data_gpu, gray_data)
    drv.memcpy_htod(kernel_gpu, kernel)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    gaussian_blur(kernel_gpu, gray_data_gpu, blurred_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size)

    blurred_data = np.empty_like(gray_data)
    drv.memcpy_dtoh(blurred_data, blurred_gpu)
    return blurred_data

def run_intensity_gradient(gray_data, width, height):
    gray_data_gpu = drv.mem_alloc(gray_data.nbytes)
    magnitudes_gpu = drv.mem_alloc(gray_data.nbytes * 4)
    directions_gpu = drv.mem_alloc(gray_data.nbytes * 4)

    drv.memcpy_htod(gray_data_gpu, gray_data)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    intensity_gradient(gray_data_gpu, np.int32(width), np.int32(height), magnitudes_gpu, directions_gpu, block=block_size, grid=grid_size)

    magnitudes = np.empty((height, width), dtype=np.float32)
    directions = np.empty((height, width), dtype=np.float32)
    drv.memcpy_dtoh(magnitudes, magnitudes_gpu)
    drv.memcpy_dtoh(directions, directions_gpu)
    return magnitudes, directions

def run_hysteresis(magnitude_data, width, height):
    magnitude_gpu = drv.mem_alloc(magnitude_data.nbytes)

    drv.memcpy_htod(magnitude_gpu, magnitude_data)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    hysteresis(magnitude_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size)

    final_edges = np.empty_like(magnitude_data)
    drv.memcpy_dtoh(final_edges, magnitude_gpu)
    return final_edges

def edge_detection_pipeline(image_path):
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    height, width, _ = img_array.shape
    # Gaussian Blur
    kernel = np.array([
        1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256,
        4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
        6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256,
        4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
        1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256
    ], dtype=np.float32)
    # Grayscale conversion
    start_cuda = time.time()
    gray_img = run_grayscale(img_array, width, height)
    blurred_img = run_gaussian_blur(gray_img, kernel, width, height)
    # Intensity Gradient
    magnitudes, directions = run_intensity_gradient(blurred_img, width, height)
    # Hysteresis
    edges = run_hysteresis((magnitudes > 100).astype(np.uint8) * 255, width, height)
    end_cuda = time.time()
    exe_time = end_cuda - start_cuda
    
    return edges, exe_time

    # # Visualization
    # plt.figure(figsize=(20, 15))

    # plt.subplot(2, 3, 1)
    # plt.title("Original RGB Image")
    # plt.imshow(img_array)
    # plt.axis('off')
    # # plt.savefig("output_original.png")

    # plt.subplot(2, 3, 2)
    # plt.title("Grayscale Image")
    # plt.imshow(gray_img, cmap='gray')
    # plt.axis('off')
    # # plt.savefig("output_grayscale.png")

    # plt.subplot(2, 3, 3)
    # plt.title("Gaussian Blurred Image")
    # plt.imshow(blurred_img, cmap='gray')
    # plt.axis('off')
    # # plt.savefig("output_blurred.png")

    # plt.subplot(2, 3, 4)
    # plt.title("Gradient Magnitudes")
    # plt.imshow(magnitudes, cmap='gray')
    # plt.axis('off')
    # # plt.savefig("output_magnitudes.png")

    # plt.subplot(2, 3, 5)
    # plt.title("Gradient Directions")
    # plt.imshow(directions, cmap='hsv')
    # plt.axis('off')
    # # plt.savefig("output_directions.png")

    # plt.subplot(2, 3, 6)
    # plt.title("Final Edges after Hysteresis")
    # plt.imshow(edges, cmap='gray')
    # plt.axis('off')
    # # plt.savefig("output_edges_shared.png")

    # plt.tight_layout()
    # plt.show()

def run_grayscale(rgb_data, width, height):
    rgb_data_gpu = drv.mem_alloc(rgb_data.nbytes)
    gray_data_gpu = drv.mem_alloc(width * height)
    drv.memcpy_htod(rgb_data_gpu, rgb_data)
    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)
    grayscale(rgb_data_gpu, gray_data_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size)
    gray_data = np.empty((height, width), dtype=np.uint8)
    drv.memcpy_dtoh(gray_data, gray_data_gpu)
    return gray_data

if __name__ == "__main__":
    result,exe_time = edge_detection_pipeline("test.png")
    print("Edge-detected image shape:", result.shape, exe_time)
