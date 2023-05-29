
#include "device.h"
#include "gpu_utils.h"
#include "tensor.h"
namespace inferllm {

void* GPUDevice::allocate(size_t len) {
    auto it = m_free_memory.lower_bound(len);
    void* ptr = nullptr;
    if (it != m_free_memory.end() && it->second.size() > 0) {
        ptr = it->second.back();
        it->second.pop_back();
        if (it->second.size() < 1) {
            m_free_memory.erase(it);
        }
    } else {
        cudaMalloc(&ptr, len);
        m_alloc_memory[ptr] = len;
    }
    return ptr;
}

void GPUDevice::free_device(void* ptr) {
    INFER_ASSERT(
            m_alloc_memory.count(ptr) == 1,
            "memory is not allocated by the DeviceCPU.");
    size_t len = m_alloc_memory[ptr];
    m_free_memory[len].push_back(ptr);
}

GPUDevice::~GPUDevice() {
    for (auto it : m_free_memory) {
        for (auto ptr : it.second) {
            INFER_ASSERT(
                    m_alloc_memory.count(ptr) == 1,
                    "memory is not allocated by the DeviceCPU.");
            cudaFree(ptr);
        }
    }

    for (auto it : m_free_memory_cpu_temp) {
        for (auto ptr : it.second) {
            INFER_ASSERT(
                    m_alloc_memory.count(ptr) == 1,
                    "memory is not allocated by the DeviceCPU.");
            aligned_free(ptr);
        }
    }
}

void* GPUDevice::cpu_temp_allocate(size_t len) {
    auto it = m_free_memory_cpu_temp.lower_bound(len);
    void* ptr = nullptr;
    if (it != m_free_memory_cpu_temp.end() && it->second.size() > 0) {
        ptr = it->second.back();
        it->second.pop_back();
        if (it->second.size() < 1) {
            m_free_memory_cpu_temp.erase(it);
        }
    } else {
        ptr = aligned_alloc(len);
        m_alloc_memory_cpu_temp[ptr] = len;
    }
    return ptr;
}

void GPUDevice::copy_data_from_cpu(float* dst, float* src, size_t len) {
    cudaError_t error = cudaMemcpy(dst, src, len, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("cudaMemcpy error: %s\n", cudaGetErrorString(error));
    }
}

}  // namespace inferllm