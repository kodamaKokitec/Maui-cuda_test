<<<<<<< HEAD
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_cuda() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("CUDA Test Program\n");
    printf("================\n");
    
    // Check CUDA device
=======
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d!\n", tid);
}

int main() {
    printf("CUDA Hello World Test\n");
    
    // GPU情報の表示
>>>>>>> 714a192637bdc28463b85e4fc8f387b4f517cf83
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);
    
<<<<<<< HEAD
    if (deviceCount == 0) {
        printf("No CUDA capable devices found!\n");
        return -1;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device 0: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Global memory: %.0f MB\n", prop.totalGlobalMem / 1048576.0);
    
    // Launch simple kernel
    printf("\nLaunching CUDA kernel...\n");
    hello_cuda<<<1, 5>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    printf("CUDA test completed successfully!\n");
    return 0;
}
=======
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    
    // カーネル実行
    printf("\nExecuting CUDA kernel...\n");
    hello_kernel<<<2, 4>>>();
    
    // デバイス同期
    cudaDeviceSynchronize();
    
    printf("\nCUDA test completed successfully!\n");
    return 0;
}
>>>>>>> 714a192637bdc28463b85e4fc8f387b4f517cf83
