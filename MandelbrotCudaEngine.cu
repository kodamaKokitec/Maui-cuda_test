// MandelbrotCudaEngine.cu - 高精度・タイル化対応版
#include "MandelbrotCudaEngine.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// ダブル精度Mandelbrot計算カーネル
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

// タイル演算用CUDAカーネル（ダブル精度）
__global__ void mandelbrot_tile_kernel_double(
    unsigned char* image, 
    int width, int height,
    double center_x, double center_y, 
    double zoom, int max_iter) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        // 高精度座標計算
        double pixel_size = 1.0 / zoom;
        double x = center_x + (idx - width / 2.0) * pixel_size;
        double y = center_y + (height / 2.0 - idy) * pixel_size;
        
        int iter = mandelbrot_double(x, y, max_iter);
        
        // カラーマッピング
        int pixel_idx = (idy * width + idx) * 4; // RGBA
        if (iter == max_iter) {
            // 集合内部
            image[pixel_idx] = 0;     // R
            image[pixel_idx + 1] = 0; // G
            image[pixel_idx + 2] = 0; // B
            image[pixel_idx + 3] = 255; // A
        } else {
            // 集合外部 - HSVベースのカラーリング
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

// C++/CLI実装
array<Byte>^ MandelbrotCudaEngine::ComputeTileDouble(
    double centerX, double centerY, 
    double zoom, int width, int height, 
    int maxIterations) {
    
    if (!isInitialized) {
        InitializeCuda();
    }
    
    // GPU メモリ確保
    size_t image_size = width * height * 4; // RGBA
    unsigned char* d_image;
    cudaMalloc((void**)&d_image, image_size);
    
    // カーネル実行設定
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    // CUDA カーネル実行
    mandelbrot_tile_kernel_double<<<gridSize, blockSize>>>(
        d_image, width, height, centerX, centerY, zoom, maxIterations);
    
    // 結果をホストにコピー
    array<Byte>^ result = gcnew array<Byte>(image_size);
    pin_ptr<Byte> pinnedResult = &result[0];
    cudaMemcpy(pinnedResult, d_image, image_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_image);
    
    return result;
}

void MandelbrotCudaEngine::InitializeCuda() {
    // CUDA デバイス初期化
    cudaSetDevice(0);
    isInitialized = true;
}
