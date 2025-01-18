import time
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

__global__ void gaussianBlur_warp(float* kernel, unsigned char* source, unsigned char* target, int width, int height) {
    __shared__ unsigned char sharedMem[16 + 4][16 + 4]; // Each block is allocated shared memory.

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;  // Ensure the thread is within bounds

    int sharedX = threadIdx.x + 2;
    int sharedY = threadIdx.y + 2;

    // Load data into shared memory (burst access)
    if (x < width && y < height) {
        sharedMem[sharedY][sharedX] = source[y * width + x];

        // Load halo data (boundary section)
        if (threadIdx.x < 2) { // Left boundary
            sharedMem[sharedY][sharedX - 2] = (x >= 2) ? source[y * width + x - 2] : 0;
        }
        if (threadIdx.x >= blockDim.x - 2) { // Right boundary
            sharedMem[sharedY][sharedX + 2] = (x + 2 < width) ? source[y * width + x + 2] : 0;
        }
        if (threadIdx.y < 2) { // Upper boundary
            sharedMem[sharedY - 2][sharedX] = (y >= 2) ? source[(y - 2) * width + x] : 0;
        }
        if (threadIdx.y >= blockDim.y - 2) { // Lower boundary
            sharedMem[sharedY + 2][sharedX] = (y + 2 < height) ? source[(y + 2) * width + x] : 0;
        }
    }
    __syncthreads(); // Ensure all threads are loaded.

    // Calculate Gaussian blur
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

__global__ void grayscale_warp(unsigned char* rgbData, unsigned char* grayData, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;  // Ensure the thread is within bounds


    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // RGB data is stored in order.
        unsigned char r = rgbData[idx];
        unsigned char g = rgbData[idx + 1];
        unsigned char b = rgbData[idx + 2];

        // Weighted average
        grayData[y * width + x] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

__global__ void intensityGradient_warp(unsigned char* img, int width, int height, float* magnitudes, float* directions) {
    __shared__ unsigned char sharedMem[16 + 2][16 + 2]; // Each thread block 16x16 + boundary

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;  // Ensure the thread is within bounds


    int sharedX = threadIdx.x + 1;
    int sharedY = threadIdx.y + 1;

    // Load data into shared memory
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
    __syncthreads(); // Ensure all threads are loaded.

    if (x < width - 1 && y < height - 1 && x > 0 && y > 0) {
        int Gx = sharedMem[sharedY - 1][sharedX + 1] + 2 * sharedMem[sharedY][sharedX + 1] + sharedMem[sharedY + 1][sharedX + 1]
               - sharedMem[sharedY - 1][sharedX - 1] - 2 * sharedMem[sharedY][sharedX - 1] - sharedMem[sharedY + 1][sharedX - 1];
        int Gy = sharedMem[sharedY + 1][sharedX - 1] + 2 * sharedMem[sharedY + 1][sharedX] + sharedMem[sharedY + 1][sharedX + 1]
               - sharedMem[sharedY - 1][sharedX - 1] - 2 * sharedMem[sharedY - 1][sharedX] - sharedMem[sharedY - 1][sharedX + 1];

        magnitudes[y * width + x] = sqrtf((float)Gx * Gx + Gy * Gy);
        float direction = atan2f((float)Gy, (float)Gx) * 180.0f / M_PI;
        directions[y * width + x] = (direction < 0) ? direction + 180.0f : direction;
    }
}

__global__ void hysteresis_warp(unsigned char* img, int width, int height) {
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

__global__ void non_maximum_suppression(const float *gradient, const float *direction, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int idx = y * width + x;
    float grad = gradient[idx];
    float angle = direction[idx];

    float neighbor1 = 0.0, neighbor2 = 0.0;

    if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
        neighbor1 = gradient[idx - 1];   // Left
        neighbor2 = gradient[idx + 1];   // Right
    } else if (angle >= 22.5 && angle < 67.5) {
        neighbor1 = gradient[idx - width + 1];  // Top-right
        neighbor2 = gradient[idx + width - 1];  // Bottom-left
    } else if (angle >= 67.5 && angle < 112.5) {
        neighbor1 = gradient[idx - width];  // Top
        neighbor2 = gradient[idx + width];  // Bottom
    } else if (angle >= 112.5 && angle < 157.5) {
        neighbor1 = gradient[idx - width - 1];  // Top-left
        neighbor2 = gradient[idx + width + 1];  // Bottom-right
    }

    if (grad >= neighbor1 && grad >= neighbor2) {
        output[idx] = grad;
    } else {
        output[idx] = 0.0;
    }
}


__global__ void convolution(float *img, float *kernel, float *output, int width, int height, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_ksize = ksize / 2;

    if (x < width && y < height) {
        float sum = 0.0;

        for (int ky = -half_ksize; ky <= half_ksize; ++ky) {
            for (int kx = -half_ksize; kx <= half_ksize; ++kx) {
                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);
                sum += img[ny * width + nx] * kernel[(ky + half_ksize) * ksize + (kx + half_ksize)];
            }
        }
        output[y * width + x] = sum;
    }
}


__global__ void hysteresis_thresholding(float *img, float *output, int width, int height, float low_thresh, float high_thresh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float pixel = img[idx];

        // Strong edges
        if (pixel >= high_thresh) {
            output[idx] = 255.0f; // Strong edge
        }
        // Weak edges
        else if (pixel >= low_thresh) {
            output[idx] = 128.0f; // Potential edge
        }
        // Non-edges
        else {
            output[idx] = 0.0f; // Suppress pixel
        }
    }
}

__global__ void compute_gradient(float *Gx, float *Gy, float *magnitude, float *direction, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        float gx = Gx[idx];
        float gy = Gy[idx];

        // Gradient Magnitude
        magnitude[idx] = sqrtf(gx * gx + gy * gy);

        // Gradient Direction (in degrees, adjusted to 0-180 range)
        direction[idx] = atan2f(gy, gx) * 180.0f / M_PI;
        if (direction[idx] < 0) direction[idx] += 180.0f;
    }
}

"""


# Compile CUDA code
mod = SourceModule(cuda_code)

# Retrieve kernels
gaussian_blur = mod.get_function("gaussianBlur_warp")
grayscale_warp = mod.get_function("grayscale_warp")
hysteresis_warp = mod.get_function("hysteresis_warp")
intensity_gradient = mod.get_function("intensityGradient_warp")


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

    
def run_hysteresis_thresholding(img, width, height, stream, low_thresh=50.0, high_thresh=100.0):
    """
    Perform hysteresis thresholding on an image.
    Args:
        img: Input image array (gradient magnitude).
        width: Width of the image.
        height: Height of the image.
        stream: CUDA stream object.
        low_thresh: Lower threshold for weak edges.
        high_thresh: Upper threshold for strong edges.
    Returns:
        Thresholded image.
    """
    img = img.astype(np.float32)
    output = np.zeros_like(img, dtype=np.float32)

    img_gpu = drv.mem_alloc(img.nbytes)
    output_gpu = drv.mem_alloc(output.nbytes)

    # Transfer data to GPU
    drv.memcpy_htod_async(img_gpu, img, stream)

    # Launch kernel
    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    #mod = SourceModule(cuda_hysteresis)
    hysteresis_kernel = mod.get_function("hysteresis_thresholding")

    hysteresis_kernel(img_gpu, output_gpu, np.int32(width), np.int32(height),
                      np.float32(low_thresh), np.float32(high_thresh),
                      block=block_size, grid=grid_size, stream=stream)

    # Retrieve results
    drv.memcpy_dtoh_async(output, output_gpu, stream)
    stream.synchronize()

    return output

def run_gradient_visualization(Gx, Gy, width, height, stream):
    """
    Compute and visualize gradient magnitudes and directions.
    Args:
        Gx: Gradient in the X-direction.
        Gy: Gradient in the Y-direction.
        width: Image width.
        height: Image height.
        stream: CUDA stream.
    Returns:
        magnitude: Gradient magnitude array.
        direction: Gradient direction array.
    """
    Gx = Gx.astype(np.float32)
    Gy = Gy.astype(np.float32)

    magnitude = np.zeros_like(Gx, dtype=np.float32)
    direction = np.zeros_like(Gx, dtype=np.float32)

    Gx_gpu = drv.mem_alloc(Gx.nbytes)
    Gy_gpu = drv.mem_alloc(Gy.nbytes)
    magnitude_gpu = drv.mem_alloc(magnitude.nbytes)
    direction_gpu = drv.mem_alloc(direction.nbytes)

    # Transfer data to GPU
    drv.memcpy_htod_async(Gx_gpu, Gx, stream)
    drv.memcpy_htod_async(Gy_gpu, Gy, stream)

    # Launch the CUDA kernel
    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    # mod = SourceModule(cuda_code_gradient)
    gradient_kernel = mod.get_function("compute_gradient")

    gradient_kernel(Gx_gpu, Gy_gpu, magnitude_gpu, direction_gpu,
                    np.int32(width), np.int32(height),
                    block=block_size, grid=grid_size, stream=stream)

    # Retrieve the results
    drv.memcpy_dtoh_async(magnitude, magnitude_gpu, stream)
    drv.memcpy_dtoh_async(direction, direction_gpu, stream)
    stream.synchronize()

    return magnitude, direction



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


def run_hysteresis_warp(magnitude_data, width, height, stream):
    magnitude_gpu = drv.mem_alloc(magnitude_data.nbytes)

    drv.memcpy_htod_async(magnitude_gpu, magnitude_data, stream)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    # grid_size = ((width + block_size[0] - 1) // block_size[0],
    #          (height + block_size[1] - 1) // block_size[1])

    grid_x = (width + block_size[0] - 1) // block_size[0]
    grid_y = (height + block_size[1] - 1) // block_size[1]
    grid_size = (grid_x, grid_y)


    # print("Grid size:", grid_size)
    # print("Block size:", block_size)


    hysteresis_warp(magnitude_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size, stream=stream)

    final_edges = np.empty_like(magnitude_data)
    drv.memcpy_dtoh_async(final_edges, magnitude_gpu, stream)
    stream.synchronize()
    return final_edges


def run_grayscale_warp(rgb_data, width, height, stream):
    rgb_data_gpu = drv.mem_alloc(rgb_data.nbytes)
    gray_data_gpu = drv.mem_alloc(width * height)
    drv.memcpy_htod_async(rgb_data_gpu, rgb_data, stream)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    grayscale_warp(rgb_data_gpu, gray_data_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size, stream=stream)

    gray_data = np.empty((height, width), dtype=np.uint8)
    drv.memcpy_dtoh_async(gray_data, gray_data_gpu, stream)
    stream.synchronize()
    return gray_data



def run_non_maximum_suppression(gradient, direction, width, height, stream):
    gradient_gpu = drv.mem_alloc(gradient.nbytes)
    direction_gpu = drv.mem_alloc(direction.nbytes)
    output_gpu = drv.mem_alloc(gradient.nbytes)

    drv.memcpy_htod_async(gradient_gpu, gradient, stream)
    drv.memcpy_htod_async(direction_gpu, direction, stream)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    non_max_suppression = mod.get_function("non_maximum_suppression")
    non_max_suppression(gradient_gpu, direction_gpu, output_gpu, np.int32(width), np.int32(height),
                        block=block_size, grid=grid_size, stream=stream)

    output = np.empty_like(gradient)
    drv.memcpy_dtoh_async(output, output_gpu, stream)
    stream.synchronize()

    return output

def run_convolution(img, kernel, width, height, stream):
    """
    Perform convolution on a grayscale image using a kernel on the GPU.
    Args:
        img: Input image array (grayscale).
        kernel: Kernel for convolution (2D array).
        width: Width of the image.
        height: Height of the image.
        stream: CUDA stream object.
    Returns:
        Convolved image.
    """
    # Convert inputs to GPU-friendly formats
    img = img.astype(np.float32)
    kernel = kernel.astype(np.float32)
    
    img_gpu = drv.mem_alloc(img.nbytes)
    kernel_gpu = drv.mem_alloc(kernel.nbytes)
    output_gpu = drv.mem_alloc(img.nbytes)

    # Transfer data to GPU
    drv.memcpy_htod_async(img_gpu, img, stream)
    drv.memcpy_htod_async(kernel_gpu, kernel, stream)

    # Launch kernel
    ksize = kernel.shape[0]
    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)
    
   # mod = SourceModule(cuda_convolution)
    convolution_kernel = mod.get_function("convolution")

    convolution_kernel(img_gpu, kernel_gpu, output_gpu, 
                       np.int32(width), np.int32(height), np.int32(ksize),
                       block=block_size, grid=grid_size, stream=stream)

    # Retrieve results
    output = np.empty_like(img)
    drv.memcpy_dtoh_async(output, output_gpu, stream)
    stream.synchronize()

    return output


def edge_detection_pipeline(image_path):
    img = Image.open(image_path).convert("RGB")  # Open and convert image to RGB
    img_array = np.array(img, dtype=np.uint8)    # Convert to NumPy array

    # print("Image shape:", img_array.shape)       # Print image shape
    # print("Unique values in input image:", np.unique(img_array))

    # Extract width and height
    height, width, _ = img_array.shape

    # Pin memory for CUDA
    pinned_img_array = drv.pagelocked_empty_like(img_array)

    # Copy input image into pinned memory
    pinned_img_array[:] = img_array

    # CUDA Stream
    stream = drv.Stream()
    
    # Gaussian Blur
    kernel = np.array([
        1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256,
        4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
        6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256,
        4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
        1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256
    ], dtype=np.float32)
    start_cuda = time.time()
    # Grayscale conversion using warp optimization
    gray_img = run_grayscale_warp(pinned_img_array, width, height, stream)
    blurred_img = run_gaussian_blur(gray_img, kernel, width,height, stream)
    # Intensity Gradient
    magnitudes, directions = run_intensity_gradient(blurred_img, width, height,stream)
    # Hysteresis
    edges = run_hysteresis_warp((magnitudes > 100).astype(np.uint8) * 255, width, height, stream)
    # blurred_img = run_gaussian_blur(gray_img, kernel, width, height, stream)

    # # Define Sobel filters
    # sobel_x = np.array([[-1, 0, 1],
    #                     [-2, 0, 2],
    #                     [-1, 0, 1]], dtype=np.float32)
    
    # sobel_y = np.array([[-1, -2, -1],
    #                     [ 0,  0,  0],
    #                     [ 1,  2,  1]], dtype=np.float32)
    
    # # Compute gradients using Sobel filters
    # gradient_x = run_convolution(gray_img, sobel_x, width, height, stream)
    # gradient_y = run_convolution(gray_img, sobel_y, width, height, stream)
    # # Compute gradient magnitude and direction
    # gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    # gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)
    # gradient_direction = np.abs(gradient_direction) % 180  # Normalize to [0, 180] degrees

    # suppressed_img = run_non_maximum_suppression(gradient_magnitude, gradient_direction, width, height, stream)
    # Hysteresis Thresholding
    
    # Perform Hysteresis Thresholding
    # thresholded_img = run_hysteresis_thresholding(suppressed_img, width, height, stream, low_thresh=50.0, high_thresh=100.0)
    end_cuda = time.time()
    exe_time = end_cuda - start_cuda    
    # Compute Gradient Magnitudes and Directions
    # gradient_magnitude, gradient_direction = run_gradient_visualization(gradient_x, gradient_y, width, height, stream)
    # = run_convolution(gray_img, sobel_x, width, height, strea
    # gradient_y = run_convolution(gray_img, sobel_y, width, height, sm)tream)
    # Plot Gradient Magnitudes
    


    # Final Output Visualization
    # plt.figure(figsize=(15, 10))
    # plt.subplot(1, 6, 1)
    # plt.title("Grayscale Image")
    # plt.imshow(gray_img, cmap="gray")

    # plt.subplot(1, 6, 2)
    # plt.title("Gaussian Blurred Image")
    # plt.imshow(blurred_img, cmap="gray")

    # plt.subplot(1, 6, 3)
    # plt.title("Non-Maximum Suppression")
    # plt.imshow(suppressed_img, cmap="gray")

    # plt.subplot(1, 6, 4)
    # plt.title("Gradient Magnitude")
    # plt.imshow(gradient_magnitude, cmap="gray")
    # # plt.axis("off")
    # # plt.colorbar()
    # # plt.show()
    
    # # Plot Gradient Directions
    # plt.subplot(1, 6, 5)
    # plt.title("Gradient Direction (Degrees)")
    # plt.imshow(gradient_direction, cmap="hsv")  # Use HSV for angle visualization
    # # plt.axis("off")
    # # plt.colorbar()
    # # plt.show()


    # plt.subplot(1, 6, 6)
    # plt.title("Hysteresis Thresholding")
    # plt.imshow(thresholded_img, cmap="gray")

    # plt.figure(figsize=(15, 10))  # Adjust the overall figure size

    # # 2x3 grid layout

    # plt.subplot(2, 3, 1)
    # plt.title("Original Image")
    # plt.imshow(img, cmap="gray")
    # plt.axis("off") 

    # plt.subplot(2, 3, 2)
    # plt.title("Grayscale Image")
    # plt.imshow(gray_img, cmap="gray")
    # plt.axis("off")  
    
    # plt.subplot(2, 3, 3)
    # plt.title("Gaussian Blurred Image")
    # plt.imshow(blurred_img, cmap="gray")
    # plt.axis("off")
    
    # # plt.subplot(2, 3, 3)
    # # plt.title("Non-Maximum Suppression")
    # # plt.imshow(suppressed_img, cmap="gray")
    # # plt.axis("off")
    
    # plt.subplot(2, 3, 4)
    # plt.title("Gradient Magnitude")
    # plt.imshow(gradient_magnitude, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(2, 3, 5)
    # plt.title("Gradient Direction (Degrees)")
    # plt.imshow(gradient_direction, cmap="hsv")  # Use HSV for angle visualization
    # plt.axis("off")
    
    # plt.subplot(2, 3, 6)
    # plt.title("Hysteresis Thresholding")
    # plt.imshow(thresholded_img, cmap="gray")
    # plt.axis("off")
    
    # # Adjust layout to avoid overlapping titles and tight spacing
    # plt.tight_layout()
    
    # plt.show()

    # plt.savefig("output_warp_final.png", dpi=300, bbox_inches="tight")  # Saves as PNG with high quality



    # plt.tight_layout()
    # plt.show()
    return edges,exe_time


if __name__ == "__main__":
    result,exe_time = edge_detection_pipeline("test.png")
    print("Edge-detected image shape:", result.shape, exe_time)