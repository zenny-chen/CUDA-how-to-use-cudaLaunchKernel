
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

static constexpr auto arraySize = 1152U;

static __global__ void addKernel(int c[], const int a[], const int b[], int constValue)
{
    auto const gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid >= arraySize) {
        return;
    }

    c[gtid] = a[gtid] * a[gtid] + (b[gtid] - constValue);
}

static void AddWithCUDATest(void)
{
    puts("======== The following is Add-With-CUDA Test ========");

    int a[arraySize];
    int b[arraySize];
    int c[arraySize] = {  };

    cudaFuncAttributes attrs{ };
    auto cudaStatus = cudaFuncGetAttributes(&attrs, addKernel);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaFuncGetAttributes call failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    auto const maxThreadCount = attrs.maxThreadsPerBlock;

    for (unsigned i = 0U; i < arraySize; ++i)
    {
        a[i] = i + 1;
        b[i] = a[i] * 10;
    }

    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    do
    {
        cudaStatus = cudaMalloc(&dev_c, sizeof(c));
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed for dev_c: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMalloc(&dev_a, sizeof(a));
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed for dev_a: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMalloc(&dev_b, sizeof(b));
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed for dev_b: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(dev_a, a, sizeof(a), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed for dev_a: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(dev_b, b, sizeof(b), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed for dev_a: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        auto const blockSize = (arraySize + maxThreadCount - 1) / maxThreadCount;

        // Launch a kernel on the GPU with one thread for each element.
        int constValue = 100;
        
        //addKernel <<< blockSize, maxThreadCount >>> (dev_c, dev_a, dev_b, constValue);
        void* args[]{ &dev_c, &dev_a, &dev_b, &constValue };
        cudaStatus = cudaLaunchKernel(addKernel, dim3(blockSize, 1U, 1U), dim3(maxThreadCount, 1U, 1U), args, 0U, nullptr);

        if (cudaStatus != cudaSuccess)
        {
            printf("cudaLaunchKernel failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed for dev_c: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }
        // Verify the result
        bool success = true;
        for (unsigned i = 0; i < arraySize; ++i)
        {
            const int correctValue = a[i] * a[i] + (b[i] - constValue);
            if (c[i] != correctValue)
            {
                printf("Result error @%u, destination is: %d, correct value is: %d\n", i, c[i], correctValue);
                success = false;
                break;
            }
        }
        if (success) {
            puts("Result is correct!");
        }
    }
    while (false);

    if (dev_a != nullptr) {
        cudaFree(dev_a);
    }
    if (dev_b != nullptr) {
        cudaFree(dev_b);
    }
    if (dev_c != nullptr) {
        cudaFree(dev_c);
    }
}

int main(int argc, const char* argv[])
{
    auto cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaSetDevice call failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    AddWithCUDATest();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaDeviceReset failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    return 0;
}
