import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from PIL import Image

# CUDA kernel code as a string
cuda_code = """
#define MY_PI 3.141592f

#define STRONG_EDGE 60
#define WEAK_EDGE 20
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

__global__ void grayscale(unsigned char* imgData, unsigned char* grayData, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.x + threadIdx.y;
    if (x >= width || y >= height) return;

    int avg = 0;
    for (int ch = 0; ch < 3; ch++) {
        int idx = y * width * 3 + x * 3 + ch;
        avg += (int)imgData[idx];
    }
    avg /= 3;
    grayData[y * width + x] = avg;
}

__global__ void intensityGradient(unsigned char* source, int width, int height, float* magnitudes, float* directions) {
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

    int totalSize = width * height;
    int Gx = 0, Gy = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int idx = (y + j) * width + x + i;
            if (idx >= 0 && idx < totalSize) {
                Gx += (int)source[idx] * kernelX[i + 1][j + 1];
                Gy += (int)source[idx] * kernelY[i + 1][j + 1];
            }
        }
    }

    int idx = y * width + x;
    float magnitude = sqrt((float)Gx * Gx + Gy * Gy);
    magnitudes[idx] = magnitude;

    float direction = atan2f(Gy, Gx);
    direction = direction * 180.0f / MY_PI;
    if (direction < 0.0f) {
        direction += 180.0f;
    }
    directions[idx] = direction;
}

__global__ void nonMaximumSuppression(float* magnitudes, float* directions, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;  // Avoid out-of-bound access

    int idx = y * width + x;
    float direction = directions[idx];

    int q = 255, r = 255;

    // Check pixel neighbors based on the gradient direction
    if ((0.0f <= direction && direction < 22.5f) || (157.5f <= direction && direction <= 180.0f)) {
        q = magnitudes[idx - 1];
        r = magnitudes[idx + 1];
    } else if (22.5f <= direction && direction < 67.5f) {
        q = magnitudes[idx - width + 1];
        r = magnitudes[idx + width - 1];
    } else if (67.5f <= direction && direction < 112.5f) {
        q = magnitudes[idx - width];
        r = magnitudes[idx + width];
    } else if (112.5f <= direction && direction < 157.5f) {
        q = magnitudes[idx - width - 1];
        r = magnitudes[idx + width + 1];
    }

    if (magnitudes[idx] >= q && magnitudes[idx] >= r) {
        output[idx] = (unsigned char)magnitudes[idx];
    } else {
        output[idx] = 0;
    }
}



__global__ void hysteresis(unsigned char* img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int totalSize = width * height;
    int idx = y * width + x;
    if (img[idx] == STRONG_EDGE) {
        img[idx] = STRONG_EDGE;  // Strong edge is retained
    }
    else if (img[idx] >= WEAK_EDGE) {
        // Check if it is connected to any strong edge
        bool connected = false;
        for (int i = -1; i <= 1 && !connected; i++) {
            for (int j = -1; j <= 1; j++) {
                int neighborIdx = (y + j) * width + x + i;
                if (connected >= 0 && neighborIdx < totalSize && img[neighborIdx] == STRONG_EDGE) {
                    connected = true;
                    break;
                }
            }
        }
        img[idx] = connected ? STRONG_EDGE : NO_EDGE;
    }
    else {
        img[idx] = NO_EDGE;  // Suppress weak edges not connected to strong ones
    }
}
"""

# Compile CUDA kernel
mod = SourceModule(cuda_code)

# Retrieve CUDA functions
grayscale = mod.get_function("grayscale")
gaussian_blur = mod.get_function("gaussianBlur")
intensity_gradient = mod.get_function("intensityGradient")
non_maximum_suppression = mod.get_function("nonMaximumSuppression")
hysteresis = mod.get_function("hysteresis")

def run_grayscale(rgb_image, width, height):
    rgb_data_gpu = drv.mem_alloc(rgb_image.nbytes)
    gray_data_gpu = drv.mem_alloc(width * height)
    drv.memcpy_htod(rgb_data_gpu, rgb_image)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    grayscale(rgb_data_gpu, gray_data_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size)

    gray_image = np.empty((height, width), dtype=np.uint8)
    drv.memcpy_dtoh(gray_image, gray_data_gpu)

    return gray_image

def run_gaussian_blur(kernel, source, width, height):
    kernel_gpu = drv.mem_alloc(kernel.nbytes)
    source_gpu = drv.mem_alloc(source.nbytes)
    target_gpu = drv.mem_alloc(source.nbytes)

    drv.memcpy_htod(kernel_gpu, kernel)
    drv.memcpy_htod(source_gpu, source)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    gaussian_blur(kernel_gpu, source_gpu, target_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size)

    blurred_image = np.empty_like(source)
    drv.memcpy_dtoh(blurred_image, target_gpu)

    return blurred_image

def run_intensity_gradient(source, width, height):
    source_gpu = drv.mem_alloc(source.nbytes)
    magnitude_gpu = drv.mem_alloc(width * height * np.dtype(np.float32).itemsize)
    direction_gpu = drv.mem_alloc(width * height * np.dtype(np.float32).itemsize)

    drv.memcpy_htod(source_gpu, source)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    magnitudes = np.empty((height, width), dtype=np.float32)
    directions = np.empty((height, width), dtype=np.float32)

    # Launch the kernel
    intensity_gradient(source_gpu, np.int32(width), np.int32(height), magnitude_gpu, direction_gpu,
                       block=block_size, grid=grid_size)

    # Synchronize to ensure kernel execution
    drv.Context.synchronize()

    # Copy back results
    drv.memcpy_dtoh(magnitudes, magnitude_gpu)
    drv.memcpy_dtoh(directions, direction_gpu)

    return magnitudes, directions

def run_non_maximum_suppression(magnitudes, directions, width, height):
    # Allocate GPU memory
    mag_gpu = drv.mem_alloc(magnitudes.nbytes)
    dir_gpu = drv.mem_alloc(directions.nbytes)
    output_gpu = drv.mem_alloc(width * height * np.dtype(np.uint8).itemsize)

    # Copy data to GPU
    drv.memcpy_htod(mag_gpu, magnitudes)
    drv.memcpy_htod(dir_gpu, directions)

    # Allocate host output
    output = np.empty((height, width), dtype=np.uint8)

    # Debugging information
    print("Magnitudes size (bytes):", magnitudes.nbytes)
    print("Directions size (bytes):", directions.nbytes)
    print("Output size (bytes):", output.nbytes)
    print("Output GPU size (bytes):", width * height * np.dtype(np.uint8).itemsize)

    # Define grid and block sizes
    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    # Run kernel
    non_maximum_suppression(mag_gpu, dir_gpu, output_gpu, np.int32(width), np.int32(height),
                            block=block_size, grid=grid_size)

    # Synchronize and check for kernel errors
    drv.Context.synchronize()

    # Copy data back to host
    drv.memcpy_dtoh(output, output_gpu)

    return output

def run_hysteresis(image, width, height):
    image_gpu = drv.mem_alloc(image.nbytes)
    drv.memcpy_htod(image_gpu, image)

    block_size = (16, 16, 1)
    grid_size = ((width + 15) // 16, (height + 15) // 16, 1)

    hysteresis(image_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size)

    result = np.empty_like(image)
    drv.memcpy_dtoh(result, image_gpu)

    return result

def main(image_path):
    # Load the image
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    rgb_image = np.array(img, dtype=np.uint8)

    # 1. Grayscale conversion
    gray_image = run_grayscale(rgb_image, width, height)
    Image.fromarray(gray_image).save("gray_image.png")

    # 2. Gaussian Blur
    # 使用更大的高斯核
    gaussian_kernel = np.array([
        1,  4,  7,  4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1,  4,  7,  4, 1,
    ], dtype=np.float32).reshape((5, 5))
    gaussian_kernel /= gaussian_kernel.sum()
    
    blurred_image = run_gaussian_blur(gaussian_kernel, gray_image, width, height)
    Image.fromarray(blurred_image).save("blurred_image.png")

    # 3. Intensity Gradient
    magnitudes, directions = run_intensity_gradient(blurred_image, width, height)

    # 4. Non-Maximum Suppression
    edges = run_non_maximum_suppression(magnitudes, directions, width, height)
    Image.fromarray(edges).save("edges.png")

    # 5. Hysteresis Thresholding
    final_edges = run_hysteresis(edges, width, height)
    Image.fromarray(final_edges).save("final_edges.png")

    print("Canny edge detection completed. Results saved!")

if __name__ == "__main__":
    main("test.png")