#ifndef MANDELBROT_CUDA_WRAPPER_H
#define MANDELBROT_CUDA_WRAPPER_H

#ifdef BUILDING_DLL
#define CUDA_WRAPPER_API __declspec(dllexport)
#else
#define CUDA_WRAPPER_API __declspec(dllimport)
#endif

// Error codes
#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_PARAMETER -1
#define CUDA_ERROR_MEMORY_ALLOCATION -2
#define CUDA_ERROR_KERNEL_EXECUTION -3
#define CUDA_ERROR_DEVICE_SYNC -4
#define CUDA_ERROR_MEMORY_COPY -5
#define CUDA_ERROR_NO_DEVICE -6
#define CUDA_ERROR_DEVICE_QUERY -7

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