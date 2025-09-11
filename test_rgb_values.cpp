#include <stdio.h>
#include <windows.h>

typedef int (*CalculateMandelbrotCudaFunc)(unsigned char*, int, int, double, double, double, int);

int main() {
    printf("=== Detailed RGB Analysis ===\n");
    
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

    // Test with small image for detailed analysis
    int width = 16;
    int height = 16;
    unsigned char* rgbData = new unsigned char[width * height * 3];
    
    printf("Testing CUDA wrapper with %dx%d image...\n", width, height);
    printf("Parameters: center=(-0.5, 0.0), zoom=1.0, iterations=100\n\n");
    
    int result = CalculateMandelbrotCuda(rgbData, width, height, -0.5, 0.0, 1.0, 100);
    
    if (result == 0) {
        printf("Success! Detailed RGB values:\n");
        printf("Pixel | R   G   B | Color Description\n");
        printf("------|-----------|------------------\n");
        
        for (int i = 0; i < width * height; i++) {
            int index = i * 3;
            unsigned char r = rgbData[index];
            unsigned char g = rgbData[index + 1];
            unsigned char b = rgbData[index + 2];
            
            const char* color_desc = "Unknown";
            if (r == 0 && g == 0 && b == 0) {
                color_desc = "Black (inside set)";
            } else if (r == 0 && g > 200 && b == 255) {
                color_desc = "Cyan (early iterations)";
            } else if (r > 200 && g == 255 && b < 50) {
                color_desc = "Yellow (mid iterations)";
            } else if (r == 255 && g < 50 && b == 0) {
                color_desc = "Red (late iterations)";
            } else if (r == 0 && g < 200 && b == 255) {
                color_desc = "Blue (very early)";
            } else if (r < 200 && g == 255 && b > 50) {
                color_desc = "Green-Cyan (early-mid)";
            } else if (r == 255 && g > 50 && b == 0) {
                color_desc = "Orange-Red (mid-late)";
            } else {
                color_desc = "Mixed/Transition";
            }
            
            printf(" %3d  | %3d %3d %3d | %s\n", i, r, g, b, color_desc);
        }
        
        // Summary
        int black = 0, blue = 0, cyan = 0, yellow = 0, red = 0, mixed = 0;
        for (int i = 0; i < width * height; i++) {
            int index = i * 3;
            unsigned char r = rgbData[index];
            unsigned char g = rgbData[index + 1];
            unsigned char b = rgbData[index + 2];
            
            if (r == 0 && g == 0 && b == 0) {
                black++;
            } else if (b > 200 && r < 50 && g < 200) {
                blue++;
            } else if (b > 200 && g > 200) {
                cyan++;
            } else if (r > 200 && g > 200 && b < 50) {
                yellow++;
            } else if (r > 200 && g < 50 && b < 50) {
                red++;
            } else {
                mixed++;
            }
        }
        
        printf("\nColor Summary:\n");
        printf("Black: %d, Blue: %d, Cyan: %d, Yellow: %d, Red: %d, Mixed: %d\n", 
               black, blue, cyan, yellow, red, mixed);
        
        if (blue > (width * height) * 0.7) {
            printf("⚠️  WARNING: Too much blue detected!\n");
        } else {
            printf("✓ Color distribution looks reasonable\n");
        }
        
    } else {
        printf("Function failed with result: %d\n", result);
    }
    
    delete[] rgbData;
    FreeLibrary(hModule);
    return 0;
}
