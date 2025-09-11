#include <stdio.h>
#include <windows.h>

typedef int (*CalculateMandelbrotCudaFunc)(unsigned char*, int, int, double, double, double, int);

int main() {
    // Load the DLL
    HMODULE hModule = LoadLibraryA("MandelbrotCudaWrapper.dll");
    if (!hModule) {
        printf("Failed to load DLL\n");
        return 1;
    }

    // Get function pointer
    CalculateMandelbrotCudaFunc CalculateMandelbrotCuda = 
        (CalculateMandelbrotCudaFunc)GetProcAddress(hModule, "CalculateMandelbrotCuda");
    
    if (!CalculateMandelbrotCuda) {
        printf("Failed to get function pointer\n");
        FreeLibrary(hModule);
        return 1;
    }

    // Test with small image
    int width = 64;
    int height = 64;
    unsigned char* rgbData = new unsigned char[width * height * 3];
    
    printf("Testing CUDA wrapper with %dx%d image...\n", width, height);
    
    // Call the function
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
        for (int i = 0; i < width * height; i++) {
            int index = i * 3;
            unsigned char r = rgbData[index];
            unsigned char g = rgbData[index + 1];
            unsigned char b = rgbData[index + 2];
            
            if (r > g && r > b && r > 50) redCount++;
            else if (g > r && g > b && g > 50) greenCount++;
            else if (b > r && b > g && b > 50) blueCount++;
            else if (r < 50 && g < 50 && b < 50) blackCount++;
        }
        
        printf("Color distribution:\n");
        printf("  Red pixels: %d\n", redCount);
        printf("  Green pixels: %d\n", greenCount);
        printf("  Blue pixels: %d\n", blueCount);
        printf("  Black pixels: %d\n", blackCount);
        printf("  Total pixels: %d\n", width * height);
    }
    
    delete[] rgbData;
    FreeLibrary(hModule);
    return 0;
}
