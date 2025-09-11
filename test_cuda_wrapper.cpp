#include <iostream>
#include <Windows.h>

// Function pointer types
typedef int (*InitializeCudaFunc)();
typedef int (*IsCudaAvailableFunc)();
typedef void (*GetGpuInfoFunc)(char*, int);
typedef int (*CalculateMandelbrotCudaFunc)(unsigned char*, int, int, double, double, double, int);
typedef void (*CleanupCudaFunc)();

int main()
{
    std::cout << "CUDA Wrapper DLL Test Program" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // Load the DLL
    HMODULE hDLL = LoadLibraryA("MandelbrotCudaWrapper.dll");
    if (!hDLL) {
        std::cout << "Error: Could not load MandelbrotCudaWrapper.dll" << std::endl;
        std::cout << "Error code: " << GetLastError() << std::endl;
        return 1;
    }
    
    std::cout << "DLL loaded successfully!" << std::endl;
    
    // Get function pointers
    auto IsCudaAvailable = (IsCudaAvailableFunc)GetProcAddress(hDLL, "IsCudaAvailable");
    auto InitializeCuda = (InitializeCudaFunc)GetProcAddress(hDLL, "InitializeCuda");
    auto GetGpuInfo = (GetGpuInfoFunc)GetProcAddress(hDLL, "GetGpuInfo");
    auto CalculateMandelbrotCuda = (CalculateMandelbrotCudaFunc)GetProcAddress(hDLL, "CalculateMandelbrotCuda");
    auto CleanupCuda = (CleanupCudaFunc)GetProcAddress(hDLL, "CleanupCuda");
    
    if (!IsCudaAvailable || !InitializeCuda || !GetGpuInfo || !CalculateMandelbrotCuda || !CleanupCuda) {
        std::cout << "Error: Could not get function pointers from DLL" << std::endl;
        FreeLibrary(hDLL);
        return 1;
    }
    
    std::cout << "Function pointers obtained successfully!" << std::endl;
    
    // Test CUDA availability
    int cudaAvailable = IsCudaAvailable();
    std::cout << "CUDA Available: " << (cudaAvailable ? "Yes" : "No") << std::endl;
    
    if (cudaAvailable) {
        // Initialize CUDA
        int initResult = InitializeCuda();
        std::cout << "CUDA Initialization: " << (initResult == 0 ? "Success" : "Failed") << std::endl;
        
        if (initResult == 0) {
            // Get GPU info
            char gpuInfo[256] = {0};
            GetGpuInfo(gpuInfo, sizeof(gpuInfo));
            std::cout << "GPU Info: " << gpuInfo << std::endl;
            
            // Test small Mandelbrot calculation
            const int testSize = 256;
            auto* testImage = new unsigned char[testSize * testSize * 3];
            
            std::cout << "Testing Mandelbrot calculation..." << std::endl;
            int calcResult = CalculateMandelbrotCuda(testImage, testSize, testSize, -0.5, 0.0, 1.0, 100);
            
            if (calcResult == 0) {
                std::cout << "Mandelbrot calculation: Success" << std::endl;
                
                // Check if image has reasonable data
                int coloredPixels = 0;
                for (int i = 0; i < testSize * testSize * 3; i += 3) {
                    if (testImage[i] > 0 || testImage[i+1] > 0 || testImage[i+2] > 0) {
                        coloredPixels++;
                    }
                }
                std::cout << "Colored pixels: " << coloredPixels << " / " << (testSize * testSize) << std::endl;
            } else {
                std::cout << "Mandelbrot calculation: Failed (error code: " << calcResult << ")" << std::endl;
            }
            
            delete[] testImage;
            
            // Cleanup
            CleanupCuda();
            std::cout << "CUDA cleanup completed." << std::endl;
        }
    }
    
    // Free the DLL
    FreeLibrary(hDLL);
    std::cout << "DLL unloaded." << std::endl;
    std::cout << "\nTest completed successfully!" << std::endl;
    
    return 0;
}
