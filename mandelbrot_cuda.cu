#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// 逕ｻ蜒上し繧､繧ｺ縺ｮ螳夂ｾｩ
#define WIDTH 1024
#define HEIGHT 1024
#define MAX_ITER 1000

// Mandelbrot髮・粋縺ｮ險育ｮ励ｒ陦後≧CUDA繧ｫ繝ｼ繝阪Ν
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

// CUDA繧ｫ繝ｼ繝阪Ν髢｢謨ｰ
__global__ void mandelbrot_kernel(unsigned char* image, int width, int height, 
                                  float x_min, float x_max, float y_min, float y_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        // 繝斐け繧ｻ繝ｫ蠎ｧ讓吶ｒ隍・ｴ蟷ｳ髱｢蠎ｧ讓吶↓螟画鋤
        float x = x_min + (x_max - x_min) * idx / (float)width;
        float y = y_min + (y_max - y_min) * idy / (float)height;
        
        // Mandelbrot髮・粋縺ｮ險育ｮ・
        int iter = mandelbrot(x, y);
        
        // 濶ｲ縺ｮ險育ｮ暦ｼ医げ繝ｬ繝ｼ繧ｹ繧ｱ繝ｼ繝ｫ・・
        unsigned char color;
        if (iter == MAX_ITER) {
            color = 0; // 鮟抵ｼ磯寔蜷亥・驛ｨ・・
        } else {
            // 繧ｫ繝ｩ繝輔Ν縺ｪ逹濶ｲ
            color = (unsigned char)((iter * 255) / MAX_ITER);
        }
        
        // RGB蛟､繧定ｨｭ螳夲ｼ亥酔縺伜､縺ｧ繧ｰ繝ｬ繝ｼ繧ｹ繧ｱ繝ｼ繝ｫ縲√∪縺溘・繧ｫ繝ｩ繝ｼ・・
        int pixel_idx = (idy * width + idx) * 3;
        if (iter == MAX_ITER) {
            // 髮・粋蜀・Κ縺ｯ鮟・
            image[pixel_idx] = 0;     // R
            image[pixel_idx + 1] = 0; // G
            image[pixel_idx + 2] = 0; // B
        } else {
            // 髮・粋螟夜Κ縺ｯ繧ｫ繝ｩ繝輔Ν縺ｫ
            float ratio = (float)iter / MAX_ITER;
            image[pixel_idx] = (unsigned char)(255 * (1.0f - ratio));     // R
            image[pixel_idx + 1] = (unsigned char)(255 * ratio * 0.5f);   // G
            image[pixel_idx + 2] = (unsigned char)(255 * ratio);          // B
        }
    }
}

// GPU諠・ｱ繧定｡ｨ遉ｺ縺吶ｋ髢｢謨ｰ
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

// PPM逕ｻ蜒上ヵ繧｡繧､繝ｫ繧剃ｿ晏ｭ倥☆繧矩未謨ｰ
void savePPM(const char* filename, unsigned char* image, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Cannot create output file %s\n", filename);
        return;
    }
    
    // PPM繝倥ャ繝繝ｼ繧呈嶌縺崎ｾｼ縺ｿ
    fprintf(file, "P6\n%d %d\n255\n", width, height);
    
    // 逕ｻ蜒上ョ繝ｼ繧ｿ繧呈嶌縺崎ｾｼ縺ｿ
    fwrite(image, 3, width * height, file);
    fclose(file);
    
    printf("Image saved as %s\n", filename);
}

// 繝｡繝｢繝ｪ菴ｿ逕ｨ驥上ｒ陦ｨ遉ｺ縺吶ｋ髢｢謨ｰ
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
    
    // GPU諠・ｱ繧定｡ｨ遉ｺ
    printGPUInfo();
    
    // 蛻晄悄繝｡繝｢繝ｪ菴ｿ逕ｨ驥上ｒ陦ｨ遉ｺ
    printf("Initial ");
    printMemoryUsage();
    
    // 繝帙せ繝亥・繝｡繝｢繝ｪ縺ｮ遒ｺ菫・
    size_t image_size = WIDTH * HEIGHT * 3; // RGB
    unsigned char* h_image = (unsigned char*)malloc(image_size);
    if (!h_image) {
        printf("Error: Cannot allocate host memory\n");
        return -1;
    }
    
    // 繝・ヰ繧､繧ｹ蛛ｴ繝｡繝｢繝ｪ縺ｮ遒ｺ菫・
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
    
    // Mandelbrot髮・粋縺ｮ陦ｨ遉ｺ遽・峇繧定ｨｭ螳・
    float x_min = -2.5f, x_max = 1.0f;
    float y_min = -1.25f, y_max = 1.25f;
    
    // CUDA繧ｫ繝ｼ繝阪Ν縺ｮ螳溯｡瑚ｨｭ螳・
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, 
                  (HEIGHT + blockSize.y - 1) / blockSize.y);
    
    printf("Grid size: %dx%d blocks\n", gridSize.x, gridSize.y);
    printf("Block size: %dx%d threads\n", blockSize.x, blockSize.y);
    printf("Total threads: %d\n", gridSize.x * gridSize.y * blockSize.x * blockSize.y);
    printf("Image resolution: %dx%d pixels\n\n", WIDTH, HEIGHT);
    
    // 螳溯｡梧凾髢捺ｸｬ螳夐幕蟋・
    clock_t start_time = clock();
    
    // CUDA Events for GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    
    printf("Starting GPU computation...\n");
    cudaEventRecord(start_gpu);
    
    // CUDA繧ｫ繝ｼ繝阪Ν繧貞ｮ溯｡・
    mandelbrot_kernel<<<gridSize, blockSize>>>(d_image, WIDTH, HEIGHT, x_min, x_max, y_min, y_max);
    
    // 繧ｫ繝ｼ繝阪Ν螳溯｡後・螳御ｺ・ｒ蠕・ｩ・
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return -1;
    }
    
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    // GPU螳溯｡梧凾髢薙ｒ險育ｮ・
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    
    printf("GPU computation completed!\n");
    printf("GPU execution time: %.2f ms\n", gpu_time);
    
    // 繝・ヰ繧､繧ｹ縺九ｉ繝帙せ繝医∈繝・・繧ｿ繧偵さ繝斐・
    printf("Copying data from GPU to CPU...\n");
    err = cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error: Memory copy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return -1;
    }
    
    // 蜈ｨ菴薙・螳溯｡梧凾髢薙ｒ險育ｮ・
    clock_t end_time = clock();
    double total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    // 逕ｻ蜒上ｒ菫晏ｭ・
    printf("Saving image...\n");
    savePPM("mandelbrot.ppm", h_image, WIDTH, HEIGHT);
    
    // 繝代ヵ繧ｩ繝ｼ繝槭Φ繧ｹ邨ｱ險医ｒ陦ｨ遉ｺ
    printf("\n=== Performance Statistics ===\n");
    printf("Total execution time: %.2f ms\n", total_time);
    printf("GPU computation time: %.2f ms\n", gpu_time);
    printf("GPU efficiency: %.1f%%\n", (gpu_time / total_time) * 100.0);
    printf("Pixels processed: %d\n", WIDTH * HEIGHT);
    printf("Processing rate: %.2f Mpixels/sec\n", (WIDTH * HEIGHT) / (gpu_time * 1000.0));
    printf("Memory bandwidth: %.2f GB/s\n", (image_size / (1024.0 * 1024.0 * 1024.0)) / (gpu_time / 1000.0));
    printf("================================\n");
    
    // 繝｡繝｢繝ｪ隗｣謾ｾ
    cudaFree(d_image);
    free(h_image);
    
    // 繧､繝吶Φ繝医ｒ蜑企勁
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    
    printf("\nProgram completed successfully!\n");
    printf("Please open 'mandelbrot.ppm' with an image viewer to see the result.\n");
    
    return 0;
}
