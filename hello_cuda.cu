#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_cuda() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("CUDA Test Program\n");
    printf("================\n");
    
    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);
    
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
