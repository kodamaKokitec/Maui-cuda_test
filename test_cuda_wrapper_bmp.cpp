#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <windows.h>

// CUDA wrapper functions
typedef int (*InitializeCudaFunc)();
typedef int (*CalculateMandelbrotCudaFunc)(unsigned char* outputBuffer, int width, int height, 
                                           double centerX, double centerY, double zoom, int maxIterations);
typedef void (*CleanupCudaFunc)();

void SaveBmpFile(const std::string& fileName, const std::vector<unsigned char>& rgbData, int width, int height)
{
    int imageSize = width * height * 3;
    int fileSize = 54 + imageSize; // BMP header is 54 bytes
    
    std::ofstream file(fileName, std::ios::binary);
    
    // BMP file header (14 bytes)
    file.put('B');
    file.put('M');
    file.write(reinterpret_cast<const char*>(&fileSize), 4);     // File size
    int reserved = 0;
    file.write(reinterpret_cast<const char*>(&reserved), 4);     // Reserved
    int offset = 54;
    file.write(reinterpret_cast<const char*>(&offset), 4);       // Offset to image data
    
    // BMP info header (40 bytes)
    int infoHeaderSize = 40;
    file.write(reinterpret_cast<const char*>(&infoHeaderSize), 4);
    file.write(reinterpret_cast<const char*>(&width), 4);        // Image width
    file.write(reinterpret_cast<const char*>(&height), 4);       // Image height
    short planes = 1;
    file.write(reinterpret_cast<const char*>(&planes), 2);       // Planes
    short bitsPerPixel = 24;
    file.write(reinterpret_cast<const char*>(&bitsPerPixel), 2); // Bits per pixel
    int compression = 0;
    file.write(reinterpret_cast<const char*>(&compression), 4);  // Compression
    file.write(reinterpret_cast<const char*>(&imageSize), 4);    // Image size
    int pixelsPerMeter = 0;
    file.write(reinterpret_cast<const char*>(&pixelsPerMeter), 4); // X pixels per meter
    file.write(reinterpret_cast<const char*>(&pixelsPerMeter), 4); // Y pixels per meter
    int colorsUsed = 0;
    file.write(reinterpret_cast<const char*>(&colorsUsed), 4);   // Colors used
    int importantColors = 0;
    file.write(reinterpret_cast<const char*>(&importantColors), 4); // Important colors
    
    // BMP data is stored bottom-to-top, so we need to flip the image
    for (int y = height - 1; y >= 0; y--)
    {
        for (int x = 0; x < width; x++)
        {
            int srcIndex = (y * width + x) * 3;
            // BMP uses BGR order, while our data is RGB
            file.put(rgbData[srcIndex + 2]); // B
            file.put(rgbData[srcIndex + 1]); // G
            file.put(rgbData[srcIndex]);     // R
        }
    }
    
    file.close();
}

int main()
{
    std::cout << "=== CUDA Wrapper BMP Test ===" << std::endl;
    
    // Load the CUDA wrapper DLL
    HMODULE cudaLib = LoadLibraryA("MandelbrotCudaWrapper.dll");
    if (!cudaLib)
    {
        std::cerr << "Failed to load MandelbrotCudaWrapper.dll" << std::endl;
        return 1;
    }
    
    // Get function pointers
    auto initFunc = (InitializeCudaFunc)GetProcAddress(cudaLib, "InitializeCuda");
    auto calcFunc = (CalculateMandelbrotCudaFunc)GetProcAddress(cudaLib, "CalculateMandelbrotCuda");
    auto cleanupFunc = (CleanupCudaFunc)GetProcAddress(cudaLib, "CleanupCuda");
    
    if (!initFunc || !calcFunc || !cleanupFunc)
    {
        std::cerr << "Failed to get function pointers" << std::endl;
        FreeLibrary(cudaLib);
        return 1;
    }
    
    // Initialize CUDA
    if (initFunc() != 0)
    {
        std::cerr << "Failed to initialize CUDA" << std::endl;
        FreeLibrary(cudaLib);
        return 1;
    }
    
    // Test parameters
    int width = 512;
    int height = 512;
    double centerX = -0.5;
    double centerY = 0.0;
    double zoom = 1.0;
    int maxIterations = 100;
    
    std::cout << "Generating " << width << "x" << height << " image..." << std::endl;
    std::cout << "Center: (" << centerX << ", " << centerY << ")" << std::endl;
    std::cout << "Zoom: " << zoom << ", Iterations: " << maxIterations << std::endl;
    
    // Allocate buffer for RGB data
    std::vector<unsigned char> rgbData(width * height * 3);
    
    // Calculate Mandelbrot set
    int result = calcFunc(rgbData.data(), width, height, centerX, centerY, zoom, maxIterations);
    
    if (result != 0)
    {
        std::cerr << "CUDA calculation failed with result: " << result << std::endl;
        cleanupFunc();
        FreeLibrary(cudaLib);
        return 1;
    }
    
    std::cout << "CUDA calculation successful!" << std::endl;
    
    // Analyze colors in the image
    int colorCounts[6] = {0}; // Black, Blue, Cyan, Green, Yellow, Red
    for (int i = 0; i < width * height; i++)
    {
        unsigned char r = rgbData[i * 3];
        unsigned char g = rgbData[i * 3 + 1];
        unsigned char b = rgbData[i * 3 + 2];
        
        if (r == 0 && g == 0 && b == 0)
            colorCounts[0]++; // Black
        else if (r < 50 && g < 50 && b > 200)
            colorCounts[1]++; // Blue
        else if (r < 50 && g > 200 && b > 200)
            colorCounts[2]++; // Cyan
        else if (r < 50 && g > 200 && b < 50)
            colorCounts[3]++; // Green
        else if (r > 200 && g > 200 && b < 50)
            colorCounts[4]++; // Yellow
        else if (r > 200 && g < 50 && b < 50)
            colorCounts[5]++; // Red
    }
    
    std::cout << "\nColor Analysis:" << std::endl;
    std::cout << "Black: " << colorCounts[0] << std::endl;
    std::cout << "Blue: " << colorCounts[1] << std::endl;
    std::cout << "Cyan: " << colorCounts[2] << std::endl;
    std::cout << "Green: " << colorCounts[3] << std::endl;
    std::cout << "Yellow: " << colorCounts[4] << std::endl;
    std::cout << "Red: " << colorCounts[5] << std::endl;
    
    // Save BMP file
    std::string fileName = "cuda_wrapper_test.bmp";
    SaveBmpFile(fileName, rgbData, width, height);
    std::cout << "\nBMP file saved as: " << fileName << std::endl;
    
    // Show first few RGB values for debugging
    std::cout << "\nFirst 10 RGB values:" << std::endl;
    for (int i = 0; i < 10 && i < width * height; i++)
    {
        std::cout << "Pixel " << i << ": R=" << (int)rgbData[i * 3] 
                  << " G=" << (int)rgbData[i * 3 + 1] 
                  << " B=" << (int)rgbData[i * 3 + 2] << std::endl;
    }
    
    // Cleanup
    cleanupFunc();
    FreeLibrary(cudaLib);
    
    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}
