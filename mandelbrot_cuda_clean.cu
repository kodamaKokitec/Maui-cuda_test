#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Image size definition
#define WIDTH 1024
#define HEIGHT 1024
#define MAX_ITER 1000

// CUDA kernel for Mandelbrot set calculation
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

// CUDA kernel function
__global__ void mandelbrot_kernel(unsigned char* image, int width, int height, 
                                  float x_min, float x_max, float y_min, float y_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        // Convert pixel coordinates to complex plane coordinates
        float x = x_min + (x_max - x_min) * idx / (float)width;
        float y = y_min + (y_max - y_min) * idy / (float)height;
        
        // Calculate Mandelbrot set
        int iter = mandelbrot(x, y);
        
        // Color calculation (grayscale)
        unsigned char color;
        if (iter == MAX_ITER) {
            color = 0; // Black (inside set)
        } else {
            // Colorful coloring
            color = (unsigned char)((iter * 255) / MAX_ITER);
        }
        
        // Set RGB values
        int pixel_idx = (idy * width + idx) * 3;
        if (iter == MAX_ITER) {
            // Inside set is black
            image[pixel_idx] = 0;     // R
            image[pixel_idx + 1] = 0; // G
            image[pixel_idx + 2] = 0; // B
        } else {
            // Outside set is colorful
            float ratio = (float)iter / MAX_ITER;
            image[pixel_idx] = (unsigned char)(255 * (1.0f - ratio));     // R
            image[pixel_idx + 1] = (unsigned char)(255 * ratio * 0.5f);   // G
            image[pixel_idx + 2] = (unsigned char)(255 * ratio);          // B
        }
    }
}

// Function to display GPU information
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

// Function to save BMP image file (Windows compatible)
void saveBMP(const char* filename, unsigned char* image, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Cannot create output file %s\n", filename);
        return;
    }
    
    // BMP file header (14 bytes)
    unsigned char bmp_file_header[14] = {
        'B', 'M',           // Signature
        0, 0, 0, 0,         // File size (will be filled)
        0, 0, 0, 0,         // Reserved
        54, 0, 0, 0         // Offset to pixel data
    };
    
    // BMP info header (40 bytes)
    unsigned char bmp_info_header[40] = {
        40, 0, 0, 0,        // Header size
        0, 0, 0, 0,         // Width (will be filled)
        0, 0, 0, 0,         // Height (will be filled)
        1, 0,               // Planes
        24, 0,              // Bits per pixel
        0, 0, 0, 0,         // Compression
        0, 0, 0, 0,         // Image size (will be filled)
        0, 0, 0, 0,         // X pixels per meter
        0, 0, 0, 0,         // Y pixels per meter
        0, 0, 0, 0,         // Colors used
        0, 0, 0, 0          // Important colors
    };
    
    // Calculate padding for 4-byte alignment
    int padding = (4 - (width * 3) % 4) % 4;
    int row_size = width * 3 + padding;
    int file_size = 54 + row_size * height;
    
    // Fill in file size
    bmp_file_header[2] = (unsigned char)(file_size);
    bmp_file_header[3] = (unsigned char)(file_size >> 8);
    bmp_file_header[4] = (unsigned char)(file_size >> 16);
    bmp_file_header[5] = (unsigned char)(file_size >> 24);
    
    // Fill in width
    bmp_info_header[4] = (unsigned char)(width);
    bmp_info_header[5] = (unsigned char)(width >> 8);
    bmp_info_header[6] = (unsigned char)(width >> 16);
    bmp_info_header[7] = (unsigned char)(width >> 24);
    
    // Fill in height
    bmp_info_header[8] = (unsigned char)(height);
    bmp_info_header[9] = (unsigned char)(height >> 8);
    bmp_info_header[10] = (unsigned char)(height >> 16);
    bmp_info_header[11] = (unsigned char)(height >> 24);
    
    // Fill in image size
    int image_size = row_size * height;
    bmp_info_header[20] = (unsigned char)(image_size);
    bmp_info_header[21] = (unsigned char)(image_size >> 8);
    bmp_info_header[22] = (unsigned char)(image_size >> 16);
    bmp_info_header[23] = (unsigned char)(image_size >> 24);
    
    // Write headers
    fwrite(bmp_file_header, 1, 14, file);
    fwrite(bmp_info_header, 1, 40, file);
    
    // Write pixel data (BMP stores bottom-to-top, BGR format)
    unsigned char padding_bytes[3] = {0, 0, 0};
    
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int src_idx = (y * width + x) * 3;
            // Convert RGB to BGR
            fputc(image[src_idx + 2], file); // B
            fputc(image[src_idx + 1], file); // G
            fputc(image[src_idx], file);     // R
        }
        // Add padding
        fwrite(padding_bytes, 1, padding, file);
    }
    
    fclose(file);
    printf("Image saved as %s\n", filename);
}

// Function to save PPM image file
void savePPM(const char* filename, unsigned char* image, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Cannot create output file %s\n", filename);
        return;
    }
    
    // Write PPM header
    fprintf(file, "P6\n%d %d\n255\n", width, height);
    
    // Write image data
    fwrite(image, 3, width * height, file);
    fclose(file);
    
    printf("Image saved as %s\n", filename);
}

// Function to display memory usage
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
    
    // Display GPU information
    printGPUInfo();
    
    // Display initial memory usage
    printf("Initial ");
    printMemoryUsage();
    
    // Allocate host memory
    size_t image_size = WIDTH * HEIGHT * 3; // RGB
    unsigned char* h_image = (unsigned char*)malloc(image_size);
    if (!h_image) {
        printf("Error: Cannot allocate host memory\n");
        return -1;
    }
    
    // Allocate device memory
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
    
    // Set Mandelbrot set display range
    float x_min = -2.5f, x_max = 1.0f;
    float y_min = -1.25f, y_max = 1.25f;
    
    // CUDA kernel execution settings
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, 
                  (HEIGHT + blockSize.y - 1) / blockSize.y);
    
    printf("Grid size: %dx%d blocks\n", gridSize.x, gridSize.y);
    printf("Block size: %dx%d threads\n", blockSize.x, blockSize.y);
    printf("Total threads: %d\n", gridSize.x * gridSize.y * blockSize.x * blockSize.y);
    printf("Image resolution: %dx%d pixels\n\n", WIDTH, HEIGHT);
    
    // Start execution time measurement
    clock_t start_time = clock();
    
    // CUDA Events for GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    
    printf("Starting GPU computation...\n");
    cudaEventRecord(start_gpu);
    
    // Execute CUDA kernel
    mandelbrot_kernel<<<gridSize, blockSize>>>(d_image, WIDTH, HEIGHT, x_min, x_max, y_min, y_max);
    
    // Wait for kernel execution to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return -1;
    }
    
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    // Calculate GPU execution time
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    
    printf("GPU computation completed!\n");
    printf("GPU execution time: %.2f ms\n", gpu_time);
    
    // Copy data from device to host
    printf("Copying data from GPU to CPU...\n");
    err = cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error: Memory copy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return -1;
    }
    
    // Calculate total execution time
    clock_t end_time = clock();
    double total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    // Save images in multiple formats
    printf("Saving images...\n");
    saveBMP("mandelbrot.bmp", h_image, WIDTH, HEIGHT);
    savePPM("mandelbrot.ppm", h_image, WIDTH, HEIGHT);
    
    // Display performance statistics
    printf("\n=== Performance Statistics ===\n");
    printf("Total execution time: %.2f ms\n", total_time);
    printf("GPU computation time: %.2f ms\n", gpu_time);
    printf("GPU efficiency: %.1f%%\n", (gpu_time / total_time) * 100.0);
    printf("Pixels processed: %d\n", WIDTH * HEIGHT);
    printf("Processing rate: %.2f Mpixels/sec\n", (WIDTH * HEIGHT) / (gpu_time * 1000.0));
    printf("Memory bandwidth: %.2f GB/s\n", (image_size / (1024.0 * 1024.0 * 1024.0)) / (gpu_time / 1000.0));
    printf("================================\n");
    
    // Free memory
    cudaFree(d_image);
    free(h_image);
    
    // Destroy events
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    
    printf("\nProgram completed successfully!\n");
    printf("Output files:\n");
    printf("  - mandelbrot.bmp (Windows compatible format)\n");
    printf("  - mandelbrot.ppm (Portable Pixmap format)\n");
    printf("\nYou can open 'mandelbrot.bmp' with Windows Photo Viewer or any image viewer.\n");
    
    return 0;
}
