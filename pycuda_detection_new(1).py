import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule

# Python helper functions
from PIL import Image
import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
# CUDA kernels
cuda_code = """
#define M_PI 3.141592f
#define STRONG_EDGE 255
#define WEAK_EDGE 128
#define NO_EDGE 0

__global__ void gaussianBlur(float* kernel, unsigned char* source, unsigned char* target, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int totalSize = width * height;
    float weightedSum = 0;
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            int idx = (y + j) * width + x + i;
            if (idx >= 0 && idx < totalSize) {
                int kernelIdx = (i + 2) * 5 + (j + 2);
                weightedSum += (int)source[idx] * kernel[kernelIdx];
            }
        }
    }

    target[y * width + x] = weightedSum;
}

__global__ void grayscale(unsigned char* rgbData, unsigned char* grayData, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int avg = 0;
    for (int ch = 0; ch < 3; ch++) {
        int idx = y * width * 3 + x * 3 + ch;
        avg += (int)rgbData[idx];
    }
    avg /= 3;

    grayData[y * width + x] = avg;
}

__global__ void hysteresis(unsigned char* img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    if (img[idx] != WEAK_EDGE) return;

    bool connected = false;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int nx = x + i;
            int ny = y + j;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int neighborIdx = ny * width + nx;
                if (img[neighborIdx] == STRONG_EDGE) {
                    connected = true;
                    break;
                }
            }
        }
        if (connected) break;
    }
    img[idx] = connected ? STRONG_EDGE : NO_EDGE;
}

__global__ void intensityGradient(unsigned char* img, int width, int height, float* magnitudes, float* directions) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int kernelX[3][3] = {
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1},
    }, kernelY[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1},
    };

    int Gx = 0, Gy = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int idx = (y + j) * width + x + i;
            if (idx >= 0 && idx < width * height) {
                Gx += (int)img[idx] * kernelX[i + 1][j + 1];
                Gy += (int)img[idx] * kernelY[i + 1][j + 1];
            }
        }
    }

    int idx = y * width + x;
    magnitudes[idx] = sqrt((float)Gx * Gx + Gy * Gy);
    float direction = atan2f(Gy, Gx) * 180.0f / M_PI;
    directions[idx] = (direction < 0.0f) ? direction + 180.0f : direction;
}
"""

# Compile CUDA code
mod = SourceModule(cuda_code)

# Retrieve kernels
gaussian_blur = mod.get_function("gaussianBlur")
grayscale = mod.get_function("grayscale")
hysteresis = mod.get_function("hysteresis")
intensity_gradient = mod.get_function("intensityGradient")


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

def visualize_all_kernels(image_path):
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    height, width, _ = img_array.shape

    # Grayscale conversion
    gray_img = run_grayscale(img_array, width, height)

    # Gaussian Blur
    kernel = np.array([
        1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256,
        4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
        6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256,
        4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
        1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256
    ], dtype=np.float32)
    blurred_img = run_gaussian_blur(gray_img, kernel, width, height)

    # Intensity Gradient
    magnitudes, directions = run_intensity_gradient(blurred_img, width, height)

    # Hysteresis
    edges = run_hysteresis((magnitudes > 100).astype(np.uint8) * 255, width, height)

    # Visualization
    plt.figure(figsize=(20, 15))

    plt.subplot(2, 3, 1)
    plt.title("Original RGB Image")
    plt.imshow(img_array)
    plt.axis('off')
    # plt.savefig("output_original.png")

    plt.subplot(2, 3, 2)
    plt.title("Grayscale Image")
    plt.imshow(gray_img, cmap='gray')
    plt.axis('off')
    # plt.savefig("output_grayscale.png")

    plt.subplot(2, 3, 3)
    plt.title("Gaussian Blurred Image")
    plt.imshow(blurred_img, cmap='gray')
    plt.axis('off')
    # plt.savefig("output_blurred.png")

    plt.subplot(2, 3, 4)
    plt.title("Gradient Magnitudes")
    plt.imshow(magnitudes, cmap='gray')
    plt.axis('off')
    # plt.savefig("output_magnitudes.png")

    plt.subplot(2, 3, 5)
    plt.title("Gradient Directions")
    plt.imshow(directions, cmap='hsv')
    plt.axis('off')
    # plt.savefig("output_directions.png")

    plt.subplot(2, 3, 6)
    plt.title("Final Edges after Hysteresis")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.savefig("output_edges.png")

    plt.tight_layout()
    plt.show()

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
    visualize_all_kernels("test.png")
