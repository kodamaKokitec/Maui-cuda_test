#define BUILDING_DLL
#include "MandelbrotCudaWrapper.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <string.h>

// CUDA kernel for Mandelbrot set calculation with double precision
__device__ int mandelbrot_device(double x, double y, int max_iter) {
    double real = x;
    double imag = y;
    int iter = 0;
    
    while (iter < max_iter && (real * real + imag * imag) <= 4.0) {
        double temp = real * real - imag * imag + x;
        imag = 2.0 * real * imag + y;
        real = temp;
        iter++;
    }
    
    return iter;
}

// HSV to RGB color conversion
__device__ void hsv_to_rgb(float h, float s, float v, unsigned char* r, unsigned char* g, unsigned char* b) {
    if (s == 0.0f) {
        *r = *g = *b = (unsigned char)(v * 255);
        return;
    }
    
    h = h * 6.0f;
    int i = (int)h;
    float f = h - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    
    switch (i) {
        case 0: *r = (unsigned char)(v * 255); *g = (unsigned char)(t * 255); *b = (unsigned char)(p * 255); break;
        case 1: *r = (unsigned char)(q * 255); *g = (unsigned char)(v * 255); *b = (unsigned char)(p * 255); break;
        case 2: *r = (unsigned char)(p * 255); *g = (unsigned char)(v * 255); *b = (unsigned char)(t * 255); break;
        case 3: *r = (unsigned char)(p * 255); *g = (unsigned char)(q * 255); *b = (unsigned char)(v * 255); break;
        case 4: *r = (unsigned char)(t * 255); *g = (unsigned char)(p * 255); *b = (unsigned char)(v * 255); break;
        default: *r = (unsigned char)(v * 255); *g = (unsigned char)(p * 255); *b = (unsigned char)(q * 255); break;
    }
}

// CUDA kernel function with improved coloring
__global__ void mandelbrot_kernel_advanced(unsigned char* image, int width, int height, 
                                           double center_x, double center_y, double zoom, int max_iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        // Calculate complex plane coordinates
        double range = 4.0 / zoom;
        double x_min = center_x - range / 2.0;
        double y_min = center_y - range / 2.0;
        
        double x = x_min + range * idx / (double)width;
        double y = y_min + range * idy / (double)height;
        
        // Calculate Mandelbrot set
        int iter = mandelbrot_device(x, y, max_iter);
        
        // Set RGB values with smooth coloring
        int pixel_idx = (idy * width + idx) * 3;
        
        if (iter == max_iter) {
            // Inside set - black
            image[pixel_idx] = 0;     // R
            image[pixel_idx + 1] = 0; // G
            image[pixel_idx + 2] = 0; // B
        } else {
            // Simple but effective color mapping based on iterations
            // Scale iteration count to create more color variety
            float scaled_iter = logf((float)iter + 1.0f) / logf((float)max_iter + 1.0f);
            
            // Create a rainbow gradient using different ranges
            float hue = scaled_iter * 360.0f; // Full color spectrum
            
            // Convert HSV to RGB (simplified)
            float c = 1.0f; // Saturation = 1
            float x = c * (1.0f - fabsf(fmodf(hue / 60.0f, 2.0f) - 1.0f));
            
            float r, g, b;
            if (hue < 60) {
                r = c; g = x; b = 0;
            } else if (hue < 120) {
                r = x; g = c; b = 0;
            } else if (hue < 180) {
                r = 0; g = c; b = x;
            } else if (hue < 240) {
                r = 0; g = x; b = c;
            } else if (hue < 300) {
                r = x; g = 0; b = c;
            } else {
                r = c; g = 0; b = x;
            }
            
            // Apply brightness based on iteration count
            float brightness = 0.5f + 0.5f * scaled_iter;
            
            image[pixel_idx] = (unsigned char)(r * brightness * 255);     // R
            image[pixel_idx + 1] = (unsigned char)(g * brightness * 255); // G
            image[pixel_idx + 2] = (unsigned char)(b * brightness * 255); // B
        }
    }
}

// Global variables for CUDA context
static bool cuda_initialized = false;
static int cuda_device_count = 0;

// Initialize CUDA runtime
extern "C" int InitializeCuda() {
    if (cuda_initialized) {
        return 0;
    }
    
    cudaError_t err = cudaGetDeviceCount(&cuda_device_count);
    if (err != cudaSuccess || cuda_device_count == 0) {
        return -1;  // No CUDA devices available
    }
    
    // Set device 0 as active
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return -2;  // Failed to set device
    }
    
    cuda_initialized = true;
    return 0;
}

// Calculate Mandelbrot set using CUDA
extern "C" int CalculateMandelbrotCuda(
    unsigned char* output_buffer,
    int width,
    int height,
    double center_x,
    double center_y,
    double zoom,
    int max_iterations
) {
    if (!cuda_initialized) {
        if (InitializeCuda() != 0) {
            return -1;  // CUDA initialization failed
        }
    }
    
    // Calculate memory size
    size_t image_size = width * height * 3;
    
    // Allocate device memory
    unsigned char* d_image;
    cudaError_t err = cudaMalloc((void**)&d_image, image_size);
    if (err != cudaSuccess) {
        return -2;  // Device memory allocation failed
    }
    
    // Set kernel execution parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Execute kernel
    mandelbrot_kernel_advanced<<<gridSize, blockSize>>>(
        d_image, width, height, center_x, center_y, zoom, max_iterations);
    
    // Wait for completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_image);
        return -3;  // Kernel execution failed
    }
    
    // Copy result to host
    err = cudaMemcpy(output_buffer, d_image, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_image);
        return -4;  // Memory copy failed
    }
    
    // Free device memory
    cudaFree(d_image);
    
    return 0;  // Success
}

// Check if CUDA is available
extern "C" int IsCudaAvailable() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

// Get GPU information
extern "C" void GetGpuInfo(char* info_buffer, int buffer_size) {
    if (!info_buffer || buffer_size <= 0) {
        return;
    }
    
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        snprintf(info_buffer, buffer_size, "Unknown GPU");
        return;
    }
    
    snprintf(info_buffer, buffer_size, "%s (Compute %d.%d, %d SMs)", 
             prop.name, prop.major, prop.minor, prop.multiProcessorCount);
}

// Generate Mandelbrot set
extern "C" DLL_EXPORT int GenerateMandelbrot(unsigned char* image_data, int width, int height, 
                                        double center_x, double center_y, double zoom, int max_iter) {
    // Validate parameters
    if (!image_data || width <= 0 || height <= 0 || zoom <= 0 || max_iter <= 0) {
        return CUDA_ERROR_INVALID_PARAMETER;
    }
    
    size_t image_size = width * height * 3 * sizeof(unsigned char);
    unsigned char* d_image = nullptr;
    
    // Allocate GPU memory
    cudaError_t err = cudaMalloc((void**)&d_image, image_size);
    if (err != cudaSuccess) {
        return CUDA_ERROR_MEMORY_ALLOCATION;
    }
    
    // Calculate CUDA execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Launch CUDA kernel
    mandelbrot_kernel_advanced<<<gridSize, blockSize>>>(d_image, width, height, center_x, center_y, zoom, max_iter);
    
    // Check for kernel execution errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_image);
        return CUDA_ERROR_KERNEL_EXECUTION;
    }
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_image);
        return CUDA_ERROR_DEVICE_SYNC;
    }
    
    // Copy result back to host
    err = cudaMemcpy(image_data, d_image, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_image);
        return CUDA_ERROR_MEMORY_COPY;
    }
    
    // Cleanup
    cudaFree(d_image);
    
    return CUDA_SUCCESS;
}

extern "C" DLL_EXPORT int GetCudaDeviceInfo(char* device_name, int name_size, int* compute_major, int* compute_minor) {
    if (!device_name || name_size <= 0 || !compute_major || !compute_minor) {
        return CUDA_ERROR_INVALID_PARAMETER;
    }
    
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        return CUDA_ERROR_NO_DEVICE;
    }
    
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return CUDA_ERROR_INVALID_PARAMETER;
    }
    
    strncpy_s(device_name, name_size, prop.name, _TRUNCATE);
    *compute_major = prop.major;
    *compute_minor = prop.minor;
    
    return CUDA_SUCCESS;
}

// Test CUDA operation
extern "C" DLL_EXPORT int TestCudaOperation() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return CUDA_ERROR_NO_DEVICE;
    }
    
    // Test simple memory allocation/deallocation
    void* test_ptr;
    err = cudaMalloc(&test_ptr, 1024);
    if (err != cudaSuccess) {
        return CUDA_ERROR_MEMORY_ALLOCATION;
    }
    
    cudaFree(test_ptr);
    return CUDA_SUCCESS;
}

// Cleanup CUDA resources
extern "C" void CleanupCuda() {
    if (cuda_initialized) {
        cudaDeviceReset();
        cuda_initialized = false;
    }
}
