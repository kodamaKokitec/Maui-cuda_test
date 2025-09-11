// MandelbrotCudaEngine.h - C++/CLI wrapper for MAUI
#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace System;
using namespace System::Runtime::InteropServices;

public ref class MandelbrotCudaEngine
{
public:
    // 高精度演算用のダブル精度版
    static array<Byte>^ ComputeTileDouble(
        double centerX, double centerY, 
        double zoom, int width, int height, 
        int maxIterations);
    
    // 標準精度演算用
    static array<Byte>^ ComputeTile(
        float centerX, float centerY, 
        float zoom, int width, int height, 
        int maxIterations);
    
    // GPU情報取得
    static String^ GetGpuInfo();
    
    // 非同期演算
    static System::Threading::Tasks::Task<array<Byte>^>^ ComputeTileAsync(
        double centerX, double centerY, 
        double zoom, int width, int height, 
        int maxIterations);

private:
    static bool isInitialized = false;
    static void InitializeCuda();
    static void CleanupCuda();
};
