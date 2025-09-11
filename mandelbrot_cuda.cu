#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// 画像サイズの定義
#define WIDTH 1024
#define HEIGHT 1024
#define MAX_ITER 1000

// Mandelbrot集合の計算を行うCUDAカーネル
__device__ int mandelbrot(float x, float y) {
    float real = x;
    float imag = y;
    int iter = 0;
    
    while (iter < MAX_ITER && (real * real + imag * imag) <= 4.0f) {
        float temp = real * real - imag * imag + x;
        imag = 2.0f * real * imag + y;
        real = temp;
        iter++;
    }
    
    return iter;
}

// CUDAカーネル関数
__global__ void mandelbrot_kernel(unsigned char* image, int width, int height, 
                                  float x_min, float x_max, float y_min, float y_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        // ピクセル座標を複素平面座標に変換
        float x = x_min + (x_max - x_min) * idx / (float)width;
        float y = y_min + (y_max - y_min) * idy / (float)height;
        
        // Mandelbrot集合の計算
        int iter = mandelbrot(x, y);
        
        // 色の計算（グレースケール）
        unsigned char color;
        if (iter == MAX_ITER) {
            color = 0; // 黒（集合内部）
        } else {
            // カラフルな着色
            color = (unsigned char)((iter * 255) / MAX_ITER);
        }
        
        // RGB値を設定（同じ値でグレースケール、またはカラー）
        int pixel_idx = (idy * width + idx) * 3;
        if (iter == MAX_ITER) {
            // 集合内部は黒
            image[pixel_idx] = 0;     // R
            image[pixel_idx + 1] = 0; // G
            image[pixel_idx + 2] = 0; // B
        } else {
            // 集合外部はカラフルに
            float ratio = (float)iter / MAX_ITER;
            image[pixel_idx] = (unsigned char)(255 * (1.0f - ratio));     // R
            image[pixel_idx + 1] = (unsigned char)(255 * ratio * 0.5f);   // G
            image[pixel_idx + 2] = (unsigned char)(255 * ratio);          // B
        }
    }
}

// GPU情報を表示する関数
void printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("=== GPU Information ===\n");
    printf("Number of CUDA devices: %d\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Memory clock rate: %.2f GHz\n", prop.memoryClockRate / 1000000.0);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
    }
    printf("========================\n\n");
}

// PPM画像ファイルを保存する関数
void savePPM(const char* filename, unsigned char* image, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Cannot create output file %s\n", filename);
        return;
    }
    
    // PPMヘッダーを書き込み
    fprintf(file, "P6\n%d %d\n255\n", width, height);
    
    // 画像データを書き込み
    fwrite(image, 3, width * height, file);
    fclose(file);
    
    printf("Image saved as %s\n", filename);
}

// メモリ使用量を表示する関数
void printMemoryUsage() {
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    
    double free_gb = free_byte / (1024.0 * 1024.0 * 1024.0);
    double total_gb = total_byte / (1024.0 * 1024.0 * 1024.0);
    double used_gb = total_gb - free_gb;
    
    printf("GPU Memory Usage:\n");
    printf("  Total: %.2f GB\n", total_gb);
    printf("  Used:  %.2f GB\n", used_gb);
    printf("  Free:  %.2f GB\n", free_gb);
    printf("  Usage: %.1f%%\n\n", (used_gb / total_gb) * 100.0);
}

int main() {
    printf("CUDA Mandelbrot Set Visualization\n");
    printf("==================================\n");
    
    // GPU情報を表示
    printGPUInfo();
    
    // 初期メモリ使用量を表示
    printf("Initial ");
    printMemoryUsage();
    
    // ホスト側メモリの確保
    size_t image_size = WIDTH * HEIGHT * 3; // RGB
    unsigned char* h_image = (unsigned char*)malloc(image_size);
    if (!h_image) {
        printf("Error: Cannot allocate host memory\n");
        return -1;
    }
    
    // デバイス側メモリの確保
    unsigned char* d_image;
    cudaError_t err = cudaMalloc((void**)&d_image, image_size);
    if (err != cudaSuccess) {
        printf("Error: Cannot allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_image);
        return -1;
    }
    
    printf("Allocated %.2f MB on GPU\n", image_size / (1024.0 * 1024.0));
    printf("After allocation ");
    printMemoryUsage();
    
    // Mandelbrot集合の表示範囲を設定
    float x_min = -2.5f, x_max = 1.0f;
    float y_min = -1.25f, y_max = 1.25f;
    
    // CUDAカーネルの実行設定
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, 
                  (HEIGHT + blockSize.y - 1) / blockSize.y);
    
    printf("Grid size: %dx%d blocks\n", gridSize.x, gridSize.y);
    printf("Block size: %dx%d threads\n", blockSize.x, blockSize.y);
    printf("Total threads: %d\n", gridSize.x * gridSize.y * blockSize.x * blockSize.y);
    printf("Image resolution: %dx%d pixels\n\n", WIDTH, HEIGHT);
    
    // 実行時間測定開始
    clock_t start_time = clock();
    
    // CUDA Events for GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    
    printf("Starting GPU computation...\n");
    cudaEventRecord(start_gpu);
    
    // CUDAカーネルを実行
    mandelbrot_kernel<<<gridSize, blockSize>>>(d_image, WIDTH, HEIGHT, x_min, x_max, y_min, y_max);
    
    // カーネル実行の完了を待機
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return -1;
    }
    
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    // GPU実行時間を計算
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    
    printf("GPU computation completed!\n");
    printf("GPU execution time: %.2f ms\n", gpu_time);
    
    // デバイスからホストへデータをコピー
    printf("Copying data from GPU to CPU...\n");
    err = cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error: Memory copy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return -1;
    }
    
    // 全体の実行時間を計算
    clock_t end_time = clock();
    double total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    // 画像を保存
    printf("Saving image...\n");
    savePPM("mandelbrot.ppm", h_image, WIDTH, HEIGHT);
    
    // パフォーマンス統計を表示
    printf("\n=== Performance Statistics ===\n");
    printf("Total execution time: %.2f ms\n", total_time);
    printf("GPU computation time: %.2f ms\n", gpu_time);
    printf("GPU efficiency: %.1f%%\n", (gpu_time / total_time) * 100.0);
    printf("Pixels processed: %d\n", WIDTH * HEIGHT);
    printf("Processing rate: %.2f Mpixels/sec\n", (WIDTH * HEIGHT) / (gpu_time * 1000.0));
    printf("Memory bandwidth: %.2f GB/s\n", (image_size / (1024.0 * 1024.0 * 1024.0)) / (gpu_time / 1000.0));
    printf("================================\n");
    
    // メモリ解放
    cudaFree(d_image);
    free(h_image);
    
    // イベントを削除
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    
    printf("\nProgram completed successfully!\n");
    printf("Please open 'mandelbrot.ppm' with an image viewer to see the result.\n");
    
    return 0;
}
