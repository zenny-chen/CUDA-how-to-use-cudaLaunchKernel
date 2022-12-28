# CUDA How To Use cudaLaunchKernel
CUDA How To Use **`cudaLaunchKernel`** to launch a kernel execution

<br />

The key point is that parameters passing should use their addresses instead of references.
```cuda
        //addKernel <<< blockSize, maxThreadCount >>> (dev_c, dev_a, dev_b, constValue);
        void* args[]{ &dev_c, &dev_a, &dev_b, &constValue };
        cudaStatus = cudaLaunchKernel(addKernel, dim3(blockSize, 1U, 1U), dim3(maxThreadCount, 1U, 1U), args, 0U, nullptr);
```

<br />

This project is built with Visual Studio 2022 Community Edition and CUDA 11.7 SDK.

