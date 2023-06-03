#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
using namespace std;

#define N  1024
// elementwise implementation copyed from https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/elementwise.cuh
constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;

inline cudaError_t GetNumBlocks(int64_t n, int *num_blocks)
{
    int dev;
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    int sm_count;
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    int tpm;
    {
        cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                     sm_count * tpm / kBlockSize * kNumWaves));
    return cudaSuccess;
}

constexpr int kMaxPackBytes = 128 / 8;
constexpr int kMaxPackSize = 8;

template <typename Function, typename... Args>
__global__ void __launch_bounds__(kBlockSize)
    ApplyGeneric(Function functor, int64_t n, float *ret, Args... args)
{

    const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
    for (int64_t i = global_tid; i < n; i += blockDim.x * gridDim.x)
    {
        ret[i] = functor(i, args...);
    }
}

template <typename Function, typename... Args>
cudaError_t LaunchKernel(Function fun, int64_t n, float *ret, Args... args)
{
    int num_blocks;
    {
        cudaError_t err = GetNumBlocks(n,&num_blocks);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    ApplyGeneric<<<num_blocks, kBlockSize>>>(fun, n, ret, args...);
    return cudaPeekAtLastError();
}

struct MultiplyFunctor
{
    __device__ float operator()(uint32_t i, float *input1, float *input2) const
    {
        return input1[i] + input2[i];
    }
};

int main()
{
    float *x_host = (float *)malloc(N * sizeof(float));
    float *x_device;
    cudaMalloc((void **)&x_device, N * sizeof(float));
    for (int i = 0; i < N; i++)
        x_host[i] = 2.0;
    cudaMemcpy(x_device, x_host, N * sizeof(float), cudaMemcpyHostToDevice);

    float *y_host = (float *)malloc(N * sizeof(float));
    float *y_device;
    cudaMalloc((void **)&y_device, N * sizeof(float));
    for (int i = 0; i < N; i++)
        y_host[i] = 2.0;
    cudaMemcpy(y_device, y_host, N * sizeof(float), cudaMemcpyHostToDevice);

    float *output_host = (float *)malloc(N * sizeof(float));
    float *output_device;
    cudaMalloc((void **)&output_device, N * sizeof(float));

    // naive elementwise
    int32_t block_num = (N + kBlockSize - 1) / kBlockSize;
    dim3 grid(block_num, 1);
    dim3 block(kBlockSize, 1);

    LaunchKernel(MultiplyFunctor(), N, output_device, x_device, y_device);
    cudaMemcpy(output_host, output_device, N * sizeof(float), cudaMemcpyDeviceToHost);

    // elementwise template

    for (int i = 0; i < N ; i++)
    {
        cout << output_host[i] << endl;
    }
    free(x_host);
    free(y_host);
    free(output_host);
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(output_device);
    return 0;
}
