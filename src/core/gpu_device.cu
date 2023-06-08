
#include "device.h"
#include "gpu_utils.cuh"
#include "tensor.h"
#include <stdio.h>
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
}

}  // namespace inferllm