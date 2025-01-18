import matplotlib.pyplot as plt
import numpy as np
from pycuda.compiler import SourceModule
from pycuda import gpuarray
from pycuda import autoinit
import pycuda.driver as drv
import pycuda.cumath as cumath
from scipy.signal import gaussian
from PIL import Image

import cupy as cp  # CuPy FFT 模块（需要安装 CuPy 依赖）

# CUDA Kernels for Grayscale Conversion and Gradient Computation
cuda_code = """
__global__ void grayscale(unsigned char* rgb, float* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    gray[y * width + x] = 0.299f * rgb[idx] + 0.587f * rgb[idx + 1] + 0.114f * rgb[idx + 2];
}

__global__ void compute_gradient(float* input, float* magnitudes, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;

        float Gx = -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)]
                   - 2.0f * input[y * width + (x - 1)] + 2.0f * input[y * width + (x + 1)]
                   - input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];

        float Gy = -input[(y - 1) * width + (x - 1)] - 2.0f * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)]
                   + input[(y + 1) * width + (x - 1)] + 2.0f * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

        magnitudes[idx] = sqrtf(Gx * Gx + Gy * Gy);
    }
}
"""

# Compile CUDA code
mod = SourceModule(cuda_code)
grayscale = mod.get_function("grayscale")
compute_gradient = mod.get_function("compute_gradient")


def fft_gaussian_blur(input_image, kernel_size=5, sigma=1.0):
    """
    Perform Gaussian blur using FFT-based convolution with CuPy.
    """
    # Create Gaussian kernel
    gauss_1d = cp.array(gaussian(kernel_size, sigma), dtype=cp.float32)
    gauss_2d = cp.outer(gauss_1d, gauss_1d)
    gauss_2d /= cp.sum(gauss_2d)

    # Pad input image and Gaussian kernel
    padded_shape = (input_image.shape[0] + kernel_size - 1, input_image.shape[1] + kernel_size - 1)
    input_image_padded = cp.zeros(padded_shape, dtype=cp.float32)
    kernel_padded = cp.zeros(padded_shape, dtype=cp.float32)

    input_image_padded[:input_image.shape[0], :input_image.shape[1]] = cp.array(input_image, dtype=cp.float32)
    kernel_padded[:kernel_size, :kernel_size] = gauss_2d

    # Perform FFT
    input_fft = cp.fft.fft2(input_image_padded)
    kernel_fft = cp.fft.fft2(kernel_padded)

    # Frequency domain multiplication
    result_fft = input_fft * kernel_fft

    # Inverse FFT to spatial domain
    result = cp.fft.ifft2(result_fft).real
    result_cropped = result[:input_image.shape[0], :input_image.shape[1]]

    return cp.asnumpy(result_cropped)

def visualize_results(image_path):
    """
    Process the input image and display results.
    """
    # Load input image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    height, width, _ = img_array.shape

    # Allocate memory for grayscale image
    gray_gpu = gpuarray.zeros((height, width), dtype=np.float32)
    rgb_gpu = gpuarray.to_gpu(img_array)

    # Grayscale conversion
    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)
    grayscale(rgb_gpu, gray_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size)
    gray_image = gray_gpu.get()

    # Gaussian Blur using FFT
    blurred_image = fft_gaussian_blur(gray_image)

    # Gradient Magnitudes
    gradient_gpu = gpuarray.zeros_like(gray_gpu)
    compute_gradient(gpuarray.to_gpu(blurred_image), gradient_gpu, np.int32(width), np.int32(height),
                     block=block_size, grid=grid_size)
    gradient_image = gradient_gpu.get()

    # Visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Grayscale Image")
    plt.imshow(gray_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Gaussian Blur (FFT)")
    plt.imshow(blurred_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Gradient Magnitudes")
    plt.imshow(gradient_image, cmap="gray")
    plt.axis("off")
    plt.savefig("final.png")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_results("test.png")
    
    