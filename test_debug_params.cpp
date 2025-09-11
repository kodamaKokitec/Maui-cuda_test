#include <stdio.h>
#include <windows.h>

typedef int (*CalculateMandelbrotCudaFunc)(unsigned char*, int, int, double, double, double, int);

int main() {
    printf("=== Debug CUDA Wrapper with Different Parameters ===\n");
    
    HMODULE hModule = LoadLibraryA(".\\MandelbrotMAUI\\Native\\MandelbrotCudaWrapper.dll");
    if (!hModule) {
        hModule = LoadLibraryA("MandelbrotCudaWrapper.dll");
        if (!hModule) {
            printf("Failed to load DLL\n");
            return 1;
        }
    }

    CalculateMandelbrotCudaFunc CalculateMandelbrotCuda = 
        (CalculateMandelbrotCudaFunc)GetProcAddress(hModule, "CalculateMandelbrotCuda");
    
    if (!CalculateMandelbrotCuda) {
        printf("Failed to get function pointer\n");
        FreeLibrary(hModule);
        return 1;
    }

    // Test with different parameters
    struct TestCase {
        const char* name;
        double center_x;
        double center_y;
        double zoom;
        int iterations;
    } tests[] = {
        {"Standard Mandelbrot view", -0.5, 0.0, 1.0, 100},
        {"Zoomed out view", -0.5, 0.0, 0.5, 100},
        {"High iteration count", -0.5, 0.0, 1.0, 1000},
        {"Different center", 0.0, 0.0, 1.0, 100}
    };
    
    for (int t = 0; t < 4; t++) {
        printf("\n=== Test %d: %s ===\n", t+1, tests[t].name);
        printf("Parameters: center=(%.1f, %.1f), zoom=%.1f, iterations=%d\n", 
               tests[t].center_x, tests[t].center_y, tests[t].zoom, tests[t].iterations);
        
        int width = 32;  // Smaller for easier debugging
        int height = 32;
        unsigned char* rgbData = new unsigned char[width * height * 3];
        
        int result = CalculateMandelbrotCuda(rgbData, width, height, 
                                           tests[t].center_x, tests[t].center_y, 
                                           tests[t].zoom, tests[t].iterations);
        
        if (result == 0) {
            // Count different pixel types
            int blackCount = 0, redCount = 0, otherCount = 0;
            int uniqueColors = 0;
            
            for (int i = 0; i < width * height; i++) {
                int index = i * 3;
                unsigned char r = rgbData[index];
                unsigned char g = rgbData[index + 1];
                unsigned char b = rgbData[index + 2];
                
                if (r == 0 && g == 0 && b == 0) {
                    blackCount++;
                } else if (r > 200 && g < 50 && b < 50) {
                    redCount++;
                } else {
                    otherCount++;
                }
            }
            
            printf("Results: Black=%d, Red=%d, Other=%d (Total=%d)\n", 
                   blackCount, redCount, otherCount, width * height);
            
            if (otherCount > redCount) {
                printf("✓ Good color variety!\n");
            } else if (blackCount > 0 && redCount < width * height * 0.8) {
                printf("~ Acceptable coloring\n");
            } else {
                printf("✗ Too much red\n");
            }
        } else {
            printf("Function failed with result: %d\n", result);
        }
        
        delete[] rgbData;
    }
    
    FreeLibrary(hModule);
    return 0;
}
