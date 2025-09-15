// MandelbrotCudaEngine.cu - 鬮倡ｲｾ蠎ｦ繝ｻ繧ｿ繧､繝ｫ蛹門ｯｾ蠢懃沿
#include "MandelbrotCudaEngine.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// 繝繝悶Ν邊ｾ蠎ｦMandelbrot險育ｮ励き繝ｼ繝阪Ν
__device__ int mandelbrot_double(double x, double y, int max_iter) {
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

// 繧ｿ繧､繝ｫ貍皮ｮ礼畑CUDA繧ｫ繝ｼ繝阪Ν・医ム繝悶Ν邊ｾ蠎ｦ・・
__global__ void mandelbrot_tile_kernel_double(
    unsigned char* image, 
    int width, int height,
    double center_x, double center_y, 
    double zoom, int max_iter) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        // 鬮倡ｲｾ蠎ｦ蠎ｧ讓呵ｨ育ｮ・
        double pixel_size = 1.0 / zoom;
        double x = center_x + (idx - width / 2.0) * pixel_size;
        double y = center_y + (height / 2.0 - idy) * pixel_size;
        
        int iter = mandelbrot_double(x, y, max_iter);
        
        // 繧ｫ繝ｩ繝ｼ繝槭ャ繝斐Φ繧ｰ
        int pixel_idx = (idy * width + idx) * 4; // RGBA
        if (iter == max_iter) {
            // 髮・粋蜀・Κ
            image[pixel_idx] = 0;     // R
            image[pixel_idx + 1] = 0; // G
            image[pixel_idx + 2] = 0; // B
            image[pixel_idx + 3] = 255; // A
        } else {
            // 髮・粋螟夜Κ - HSV繝吶・繧ｹ縺ｮ繧ｫ繝ｩ繝ｼ繝ｪ繝ｳ繧ｰ
            float hue = (float)iter / max_iter * 360.0f;
            float sat = 1.0f;
            float val = iter < max_iter ? 1.0f : 0.0f;
            
            // HSV to RGB conversion (simplified)
            float c = val * sat;
            float x_color = c * (1 - abs(fmod(hue / 60.0f, 2) - 1));
            float m = val - c;
            
            float r, g, b;
            if (hue < 60) { r = c; g = x_color; b = 0; }
            else if (hue < 120) { r = x_color; g = c; b = 0; }
            else if (hue < 180) { r = 0; g = c; b = x_color; }
            else if (hue < 240) { r = 0; g = x_color; b = c; }
            else if (hue < 300) { r = x_color; g = 0; b = c; }
            else { r = c; g = 0; b = x_color; }
            
            image[pixel_idx] = (unsigned char)((r + m) * 255);
            image[pixel_idx + 1] = (unsigned char)((g + m) * 255);
            image[pixel_idx + 2] = (unsigned char)((b + m) * 255);
            image[pixel_idx + 3] = 255; // A
        }
    }
}

// C++/CLI螳溯｣・
array<Byte>^ MandelbrotCudaEngine::ComputeTileDouble(
    double centerX, double centerY, 
    double zoom, int width, int height, 
    int maxIterations) {
    
    if (!isInitialized) {
        InitializeCuda();
    }
    
    // GPU 繝｡繝｢繝ｪ遒ｺ菫・
    size_t image_size = width * height * 4; // RGBA
    unsigned char* d_image;
    cudaMalloc((void**)&d_image, image_size);
    
    // 繧ｫ繝ｼ繝阪Ν螳溯｡瑚ｨｭ螳・
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    // CUDA 繧ｫ繝ｼ繝阪Ν螳溯｡・
    mandelbrot_tile_kernel_double<<<gridSize, blockSize>>>(
        d_image, width, height, centerX, centerY, zoom, maxIterations);
    
    // 邨先棡繧偵・繧ｹ繝医↓繧ｳ繝斐・
    array<Byte>^ result = gcnew array<Byte>(image_size);
    pin_ptr<Byte> pinnedResult = &result[0];
    cudaMemcpy(pinnedResult, d_image, image_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_image);
    
    return result;
}

void MandelbrotCudaEngine::InitializeCuda() {
    // CUDA 繝・ヰ繧､繧ｹ蛻晄悄蛹・
    cudaSetDevice(0);
    isInitialized = true;
}
