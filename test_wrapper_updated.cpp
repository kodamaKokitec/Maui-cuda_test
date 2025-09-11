#include <stdio.h>
#include <windows.h>

typedef int (*CalculateMandelbrotCudaFunc)(unsigned char*, int, int, double, double, double, int);

int main() {
    printf("=== Testing Updated CUDA Wrapper ===\n");
    
    // Load the DLL from current directory
    HMODULE hModule = LoadLibraryA(".\\MandelbrotMAUI\\Native\\MandelbrotCudaWrapper.dll");
    if (!hModule) {
        printf("Failed to load DLL from Native directory\n");
        
        // Try loading from current directory
        hModule = LoadLibraryA("MandelbrotCudaWrapper.dll");
        if (!hModule) {
            printf("Failed to load DLL from current directory\n");
            return 1;
        }
    }

    // Get function pointer
    CalculateMandelbrotCudaFunc CalculateMandelbrotCuda = 
        (CalculateMandelbrotCudaFunc)GetProcAddress(hModule, "CalculateMandelbrotCuda");
    
    if (!CalculateMandelbrotCuda) {
        printf("Failed to get function pointer\n");
        FreeLibrary(hModule);
        return 1;
    }

    // Test with small image - same parameters as working version
    int width = 64;
    int height = 64;
    unsigned char* rgbData = new unsigned char[width * height * 3];
    
    printf("Testing CUDA wrapper with %dx%d image...\n", width, height);
    printf("Parameters: center=(-0.5, 0.0), zoom=1.0, iterations=100\n");
    
    // Call the function with standard Mandelbrot parameters
    int result = CalculateMandelbrotCuda(rgbData, width, height, -0.5, 0.0, 1.0, 100);
    
    printf("Function returned: %d\n", result);
    
    if (result == 0) {
        printf("Success! Analyzing RGB data:\n");
        
        // Print first 10 pixels
        for (int i = 0; i < 10 && i < width * height; i++) {
            int index = i * 3;
            printf("Pixel %d: R=%d, G=%d, B=%d\n", i, 
                   rgbData[index], rgbData[index + 1], rgbData[index + 2]);
        }
        
        // Color distribution analysis
        int redCount = 0, greenCount = 0, blueCount = 0, blackCount = 0;
        int mixedCount = 0;
        
        for (int i = 0; i < width * height; i++) {
            int index = i * 3;
            unsigned char r = rgbData[index];
            unsigned char g = rgbData[index + 1];
            unsigned char b = rgbData[index + 2];
            
            if (r == 0 && g == 0 && b == 0) {
                blackCount++;
            } else if (r > 200 && g < 50 && b < 50) {
                redCount++;
            } else if (g > 200 && r < 50 && b < 50) {
                greenCount++;
            } else if (b > 200 && r < 50 && g < 50) {
                blueCount++;
            } else {
                mixedCount++;
            }
        }
        
        printf("\nColor distribution:\n");
        printf("  Pure red pixels: %d\n", redCount);
        printf("  Pure green pixels: %d\n", greenCount);
        printf("  Pure blue pixels: %d\n", blueCount);
        printf("  Black pixels (inside set): %d\n", blackCount);
        printf("  Mixed color pixels: %d\n", mixedCount);
        printf("  Total pixels: %d\n", width * height);
        
        // Check for the expected gradient pattern
        if (mixedCount > redCount && blackCount > 0) {
            printf("\n✓ Color gradient detected - this looks like proper Mandelbrot coloring!\n");
        } else if (redCount > mixedCount * 2) {
            printf("\n✗ Still too much red - color calculation may need further adjustment\n");
        }
    }
    
    delete[] rgbData;
    FreeLibrary(hModule);
    return 0;
}
