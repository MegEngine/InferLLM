
#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <driver_types.h>
#define BLOCK_SIZE 256
#define TILE_SIZE 16

#define CUDA_CHECK(condition)                                             \
    /* Code block avoids redefinition of cudaError_t error */             \
    do {                                                                  \
        cudaError_t error = condition;                                    \
        CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)

#define CUBLAS_CHECK(condition)                                \
    do {                                                       \
        cublasStatus_t status = condition;                     \
        CHECK_EQ(status, CUBLAS_STATUS_SUCCESS)                \
                << " " << caffe::cublasGetErrorString(status); \
    } while (0)

#define CURAND_CHECK(condition)                                \
    do {                                                       \
        curandStatus_t status = condition;                     \
        CHECK_EQ(status, CURAND_STATUS_SUCCESS)                \
                << " " << caffe::curandGetErrorString(status); \
    } while (0)

// CUDA: grid stride looping
