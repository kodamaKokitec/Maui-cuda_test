#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d!\n", tid);
}

int main() {
    printf("CUDA Hello World Test\n");
    
    // GPU情報の表示
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);
    
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