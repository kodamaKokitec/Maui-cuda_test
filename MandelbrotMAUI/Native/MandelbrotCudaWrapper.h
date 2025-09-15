#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Export declarations for Windows DLL
#ifdef _WIN32
    #ifdef BUILDING_DLL
        #define DLL_EXPORT __declspec(dllexport)
    #else
        #define DLL_EXPORT __declspec(dllimport)
    #endif
#else
    #define DLL_EXPORT
#endif

// Error codes
#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_PARAMETER -1
#define CUDA_ERROR_MEMORY_ALLOCATION -2
#define CUDA_ERROR_KERNEL_LAUNCH -3
#define CUDA_ERROR_MEMORY_COPY -4
#define CUDA_ERROR_KERNEL_EXECUTION -5
#define CUDA_ERROR_DEVICE_SYNC -6
#define CUDA_ERROR_NO_DEVICE -7

// Initialize CUDA runtime
DLL_EXPORT int InitializeCuda();

// Calculate Mandelbrot set using CUDA
// Returns 0 on success, negative value on error
DLL_EXPORT int CalculateMandelbrotCuda(
    unsigned char* output_buffer,  // Output RGB buffer (width * height * 3 bytes)
    int width,                     // Image width
    int height,                    // Image height
    double center_x,               // Center X coordinate in complex plane
    double center_y,               // Center Y coordinate in complex plane
    double zoom,                   // Zoom level
    int max_iterations             // Maximum iterations
);

// Check if CUDA is available
DLL_EXPORT int IsCudaAvailable();

// Get GPU information
DLL_EXPORT void GetGpuInfo(char* info_buffer, int buffer_size);

// Cleanup CUDA resources
DLL_EXPORT void CleanupCuda();

#ifdef __cplusplus
}
#endif
#ifndef MANDELBROT_CUDA_WRAPPER_H
#define MANDELBROT_CUDA_WRAPPER_H

#ifdef BUILDING_DLL
#define CUDA_WRAPPER_API __declspec(dllexport)
#else
#define CUDA_WRAPPER_API __declspec(dllimport)
#endif

extern "C" {
    // Generate Mandelbrot set image
    // image_data: RGB buffer (width * height * 3 bytes)
    // width, height: image dimensions
    // center_x, center_y: complex plane center
    // zoom: zoom factor (1.0 = default view)
    // max_iter: maximum iterations
    CUDA_WRAPPER_API int GenerateMandelbrot(unsigned char* image_data, int width, int height, 
                                            double center_x, double center_y, double zoom, int max_iter);
    
    // Get CUDA device information
    CUDA_WRAPPER_API int GetCudaDeviceInfo(char* device_name, int name_size, int* compute_major, int* compute_minor);
    
    // Test CUDA operation
    CUDA_WRAPPER_API int TestCudaOperation();
}

#endif // MANDELBROT_CUDA_WRAPPER_H
