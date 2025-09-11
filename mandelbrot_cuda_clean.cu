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
        'B', 'M',           // BM signature
        0, 0, 0, 0,         // File size (will be filled)
        0, 0, 0, 0,         // Reserved
        54, 0, 0, 0         // Offset to pixel data
    };
    
    // BMP info header (40 bytes)
    unsigned char bmp_info_header[40] = {
        40, 0, 0, 0,        // Info header size
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
    
    // Calculate image size and file size
    int row_padding = (4 - (width * 3) % 4) % 4;
    int image_size = (width * 3 + row_padding) * height;
    int file_size = 54 + image_size;
    
    // Fill file size
    bmp_file_header[2] = file_size & 0xFF;
    bmp_file_header[3] = (file_size >> 8) & 0xFF;
    bmp_file_header[4] = (file_size >> 16) & 0xFF;
    bmp_file_header[5] = (file_size >> 24) & 0xFF;
    
    // Fill width
    bmp_info_header[4] = width & 0xFF;
    bmp_info_header[5] = (width >> 8) & 0xFF;
    bmp_info_header[6] = (width >> 16) & 0xFF;
    bmp_info_header[7] = (width >> 24) & 0xFF;
    
    // Fill height
    bmp_info_header[8] = height & 0xFF;
    bmp_info_header[9] = (height >> 8) & 0xFF;
    bmp_info_header[10] = (height >> 16) & 0xFF;
    bmp_info_header[11] = (height >> 24) & 0xFF;
    
    // Fill image size
    bmp_info_header[20] = image_size & 0xFF;
    bmp_info_header[21] = (image_size >> 8) & 0xFF;
    bmp_info_header[22] = (image_size >> 16) & 0xFF;
    bmp_info_header[23] = (image_size >> 24) & 0xFF;
    
    // Write headers
    fwrite(bmp_file_header, 1, 14, file);
    fwrite(bmp_info_header, 1, 40, file);
    
    // Write pixel data (bottom-up)
    unsigned char padding[3] = {0, 0, 0};
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            // BMP uses BGR format
            fputc(image[idx + 2], file); // B
            fputc(image[idx + 1], file); // G
            fputc(image[idx], file);     // R
        }
        // Add row padding
        fwrite(padding, 1, row_padding, file);
    }
    
    fclose(file);
    printf("BMP image saved as: %s\n", filename);
}

// Function to save PPM image file
void savePPM(const char* filename, unsigned char* image, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Cannot create output file %s\n", filename);
        return;
    }
    
    // PPM header
    fprintf(file, "P6\n%d %d\n255\n", width, height);
    
    // Write pixel data
    fwrite(image, 3, width * height, file);
    
    fclose(file);
    printf("PPM image saved as: %s\n", filename);
}

int main() {
    printf("CUDA Mandelbrot Set Visualization\n");
    printf("Image size: %dx%d pixels\n", WIDTH, HEIGHT);
    printf("Max iterations: %d\n\n", MAX_ITER);
    
    // Display GPU information
    printGPUInfo();
    
    // Memory allocation
    size_t image_size = WIDTH * HEIGHT * 3 * sizeof(unsigned char);
    unsigned char* h_image = (unsigned char*)malloc(image_size);
    unsigned char* d_image;
    
    if (!h_image) {
        printf("Error: Cannot allocate host memory\n");
        return -1;
    }
    
    // CUDA memory allocation
    cudaError_t err = cudaMalloc((void**)&d_image, image_size);
    if (err != cudaSuccess) {
        printf("Error: CUDA memory allocation failed: %s\n", cudaGetErrorString(err));
        free(h_image);
        return -1;
    }
    
    printf("Memory allocated: %.2f MB (Host + Device)\n", (image_size * 2) / (1024.0 * 1024.0));
    
    // Complex plane bounds
    float x_min = -2.5f, x_max = 1.0f;
    float y_min = -1.25f, y_max = 1.25f;
    
    printf("Complex plane bounds: [%.2f, %.2f] x [%.2f, %.2f]\n", x_min, x_max, y_min, y_max);
    
    // CUDA execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);
    
    printf("CUDA configuration: Grid(%d, %d), Block(%d, %d)\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    printf("Total threads: %d\n\n", gridSize.x * gridSize.y * blockSize.x * blockSize.y);
    
    // Performance measurement
    clock_t start_time = clock();
    
    // Create CUDA events for precise timing
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    // Record start event
    cudaEventRecord(start_event);
    
    // Launch CUDA kernel
    mandelbrot_kernel<<<gridSize, blockSize>>>(d_image, WIDTH, HEIGHT, x_min, x_max, y_min, y_max);
    
    // Record stop event
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    
    // Calculate execution time
    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start_event, stop_event);
    
    // Check for CUDA errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: CUDA kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return -1;
    }
    
    // Copy result back to host
    err = cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error: CUDA memory copy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return -1;
    }
    
    clock_t end_time = clock();
    
    // Performance statistics
    double total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    double pixels_per_second = (WIDTH * HEIGHT) / (gpu_time_ms / 1000.0);
    double megapixels_per_second = pixels_per_second / 1000000.0;
    
    printf("=== Performance Results ===\n");
    printf("GPU execution time: %.3f ms\n", gpu_time_ms);
    printf("Total execution time: %.3f s\n", total_time);
    printf("Pixels processed: %d\n", WIDTH * HEIGHT);
    printf("Performance: %.2f Mpixels/sec\n", megapixels_per_second);
    printf("===========================\n\n");
    
    // Save images
    printf("Saving output files...\n");
    saveBMP("mandelbrot.bmp", h_image, WIDTH, HEIGHT);
    savePPM("mandelbrot.ppm", h_image, WIDTH, HEIGHT);
    
    // Cleanup
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_image);
    free(h_image);
    
    printf("\nExecution completed successfully!\n");
    printf("Open 'mandelbrot.bmp' with Windows Photo Viewer to see the result.\n");
    
    return 0;
}