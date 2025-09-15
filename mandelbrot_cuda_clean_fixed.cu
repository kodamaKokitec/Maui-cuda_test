#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// CUDA kernel for Mandelbrot set calculation
__global__ void mandelbrot_kernel(unsigned char* image, int width, int height, 
                                  double min_x, double max_x, double min_y, double max_y, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Map pixel coordinates to complex plane
    double real = min_x + (double)x * (max_x - min_x) / width;
    double imag = min_y + (double)y * (max_y - min_y) / height;
    
    // Mandelbrot iteration
    double z_real = 0.0, z_imag = 0.0;
    int iter = 0;
    
    while (z_real * z_real + z_imag * z_imag <= 4.0 && iter < max_iter) {
        double temp = z_real * z_real - z_imag * z_imag + real;
        z_imag = 2.0 * z_real * z_imag + imag;
        z_real = temp;
        iter++;
    }
    
    // Color mapping
    int pixel_index = (y * width + x) * 3;
    if (iter == max_iter) {
        // Inside the set - black
        image[pixel_index] = 0;     // Blue
        image[pixel_index + 1] = 0; // Green  
        image[pixel_index + 2] = 0; // Red
    } else {
        // Outside the set - color based on iteration count
        double t = (double)iter / max_iter;
        image[pixel_index] = (unsigned char)(255 * (1 - t));     // Blue
        image[pixel_index + 1] = (unsigned char)(255 * t * 0.5); // Green
        image[pixel_index + 2] = (unsigned char)(255 * t);       // Red
    }
}

// Function to print GPU information
void printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("CUDA Device Information:\n");
    printf("========================\n");
    printf("Number of CUDA devices: %d\n", deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    }
    printf("========================\n\n");
}

// Function to save image as BMP
void saveBMP(const char* filename, unsigned char* image, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Could not create BMP file %s\n", filename);
        return;
    }
    
    // Calculate padding for 4-byte alignment
    int padding = (4 - (width * 3) % 4) % 4;
    int row_size = width * 3 + padding;
    int image_size = row_size * height;
    int file_size = 54 + image_size;
    
    // BMP file header (14 bytes)
    unsigned char bmp_file_header[14] = {
        'B', 'M',           // BM signature
        0, 0, 0, 0,         // File size (will be filled)
        0, 0, 0, 0,         // Reserved
        54, 0, 0, 0         // Offset to pixel data
    };
    
    // Fill file size
    bmp_file_header[2] = (unsigned char)(file_size);
    bmp_file_header[3] = (unsigned char)(file_size >> 8);
    bmp_file_header[4] = (unsigned char)(file_size >> 16);
    bmp_file_header[5] = (unsigned char)(file_size >> 24);
    
    // BMP info header (40 bytes)
    unsigned char bmp_info_header[40] = {
        40, 0, 0, 0,        // Header size
        0, 0, 0, 0,         // Image width (will be filled)
        0, 0, 0, 0,         // Image height (will be filled)
        1, 0,               // Color planes
        24, 0,              // Bits per pixel
        0, 0, 0, 0,         // Compression
        0, 0, 0, 0,         // Image size
        0, 0, 0, 0,         // X pixels per meter
        0, 0, 0, 0,         // Y pixels per meter
        0, 0, 0, 0,         // Colors used
        0, 0, 0, 0          // Important colors
    };
    
    // Fill width and height
    bmp_info_header[4] = (unsigned char)(width);
    bmp_info_header[5] = (unsigned char)(width >> 8);
    bmp_info_header[6] = (unsigned char)(width >> 16);
    bmp_info_header[7] = (unsigned char)(width >> 24);
    
    bmp_info_header[8] = (unsigned char)(height);
    bmp_info_header[9] = (unsigned char)(height >> 8);
    bmp_info_header[10] = (unsigned char)(height >> 16);
    bmp_info_header[11] = (unsigned char)(height >> 24);
    
    // Write headers
    fwrite(bmp_file_header, 1, 14, file);
    fwrite(bmp_info_header, 1, 40, file);
    
    // Write pixel data (bottom to top for BMP format)
    unsigned char padding_bytes[3] = {0, 0, 0};
    for (int y = height - 1; y >= 0; y--) {
        fwrite(&image[y * width * 3], 3, width, file);
        fwrite(padding_bytes, 1, padding, file);
    }
    
    fclose(file);
    printf("BMP image saved as: %s\n", filename);
}

// Function to save image as PPM
void savePPM(const char* filename, unsigned char* image, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Could not create PPM file %s\n", filename);
        return;
    }
    
    // Write PPM header
    fprintf(file, "P6\n%d %d\n255\n", width, height);
    
    // Write pixel data (RGB format, need to swap to RGB from BGR)
    for (int i = 0; i < width * height * 3; i += 3) {
        fputc(image[i + 2], file); // Red
        fputc(image[i + 1], file); // Green
        fputc(image[i], file);     // Blue
    }
    
    fclose(file);
    printf("PPM image saved as: %s\n", filename);
}

int main() {
    printf("CUDA Mandelbrot Set Visualization\n");
    printf("==================================\n");
    
    // Print GPU information
    printGPUInfo();
    
    // Image parameters
    const int width = 1024;
    const int height = 1024;
    const int max_iter = 1000;
    const int image_size = width * height * 3; // RGB
    
    // Mandelbrot set bounds
    const double min_x = -2.5;
    const double max_x = 1.0;
    const double min_y = -1.25;
    const double max_y = 1.25;
    
    printf("Generating %dx%d Mandelbrot set...\n", width, height);
    printf("Bounds: [%.2f, %.2f] x [%.2f, %.2f]\n", min_x, max_x, min_y, max_y);
    printf("Max iterations: %d\n\n", max_iter);
    
    // Allocate host memory
    unsigned char* h_image = (unsigned char*)malloc(image_size);
    if (!h_image) {
        printf("Error: Could not allocate host memory\n");
        return -1;
    }
    
    // Allocate device memory
    unsigned char* d_image;
    cudaError_t err = cudaMalloc((void**)&d_image, image_size);
    if (err != cudaSuccess) {
        printf("Error: Could not allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_image);
        return -1;
    }
    
    // Set up execution configuration
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);
    
    printf("Grid size: %dx%d blocks\n", grid_size.x, grid_size.y);
    printf("Block size: %dx%d threads\n", block_size.x, block_size.y);
    printf("Total threads: %d\n\n", grid_size.x * grid_size.y * block_size.x * block_size.y);
    
    // Start timing
    clock_t start_time = clock();
    
    // Launch kernel
    printf("Launching CUDA kernel...\n");
    mandelbrot_kernel<<<grid_size, block_size>>>(d_image, width, height, 
                                                  min_x, max_x, min_y, max_y, max_iter);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return -1;
    }
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return -1;
    }
    
    printf("Kernel execution completed successfully!\n");
    
    // Copy result back to host
    err = cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error: Could not copy data from device: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(h_image);
        return -1;
    }
    
    // End timing
    clock_t end_time = clock();
    double total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    // Performance statistics
    long long total_pixels = (long long)width * height;
    double mpixels_per_sec = (total_pixels / 1000000.0) / (total_time / 1000.0);
    
    printf("\nPerformance Statistics:\n");
    printf("=======================\n");
    printf("Total pixels: %lld\n", total_pixels);
    printf("Execution time: %.2f ms\n", total_time);
    printf("Processing speed: %.2f Mpixels/sec\n", mpixels_per_sec);
    printf("Memory throughput: %.2f MB/s\n", (image_size / 1024.0 / 1024.0) / (total_time / 1000.0));
    
    // Save images
    printf("\nSaving images...\n");
    saveBMP("mandelbrot.bmp", h_image, width, height);
    savePPM("mandelbrot.ppm", h_image, width, height);
    
    // Cleanup
    cudaFree(d_image);
    free(h_image);
    
    printf("\nCUDA Mandelbrot calculation completed successfully!\n");
    printf("You can view the generated images:\n");
    printf("- mandelbrot.bmp (Windows standard format)\n");
    printf("- mandelbrot.ppm (Portable Pixmap format)\n");
    
    return 0;
}
