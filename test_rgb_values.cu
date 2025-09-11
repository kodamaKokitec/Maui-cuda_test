#include <stdio.h>
#include <cuda_runtime.h>

// HSV to RGB conversion function
__device__ void hsv_to_rgb(float h, float s, float v, float& r, float& g, float& b) {
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    
    if (h >= 0 && h < 60) {
        r = c; g = x; b = 0;
    } else if (h >= 60 && h < 120) {
        r = x; g = c; b = 0;
    } else if (h >= 120 && h < 180) {
        r = 0; g = c; b = x;
    } else if (h >= 180 && h < 240) {
        r = 0; g = x; b = c;
    } else if (h >= 240 && h < 300) {
        r = x; g = 0; b = c;
    } else {
        r = c; g = 0; b = x;
    }
    
    r = (r + m) * 255.0f;
    g = (g + m) * 255.0f;
    b = (b + m) * 255.0f;
}

__global__ void test_kernel(unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int index = (y * width + x) * 3;
    
    // Test pattern: Different colors for different regions
    if (x < width/4) {
        // Red region
        output[index] = 255;     // R
        output[index + 1] = 0;   // G
        output[index + 2] = 0;   // B
    } else if (x < width/2) {
        // Green region
        output[index] = 0;       // R
        output[index + 1] = 255; // G
        output[index + 2] = 0;   // B
    } else if (x < 3*width/4) {
        // Blue region
        output[index] = 0;       // R
        output[index + 1] = 0;   // G
        output[index + 2] = 255; // B
    } else {
        // HSV test - various colors
        float h = (float)(y * 360) / height;
        float s = 1.0f;
        float v = 1.0f;
        
        float r, g, b;
        hsv_to_rgb(h, s, v, r, g, b);
        
        output[index] = (unsigned char)r;
        output[index + 1] = (unsigned char)g;
        output[index + 2] = (unsigned char)b;
    }
}

extern "C" __declspec(dllexport) int TestRgbOutput(unsigned char* rgbData, int width, int height) {
    // Allocate GPU memory
    unsigned char* d_output;
    size_t size = width * height * 3;
    
    cudaError_t err = cudaMalloc(&d_output, size);
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    test_kernel<<<gridSize, blockSize>>>(d_output, width, height);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        return -2;
    }
    
    // Copy result back to host
    err = cudaMemcpy(rgbData, d_output, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Memory copy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        return -3;
    }
    
    cudaFree(d_output);
    return 0;
}
