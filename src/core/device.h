#pragma once

#include <functional>
#include <map>

#include "kern/kernel.h"
#include "thread_pool.h"
#include "utils.h"

#ifdef ENABLE_GPU
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>
#endif

namespace inferllm {

class Tensor;

//! TODO: this task maybe as class to implement the CPU task and GPU task
using Task = std::function<void(void)>;

class Device {
public:
    Device(){};

    virtual void* allocate(size_t len) = 0;
    virtual void free_device(void* ptr) = 0;
    virtual ~Device() = default;

    Kernel* kernel() { return m_kernel.get(); }
    
    KernelType type() { return m_kernel->m_kernel_type; };

    void* aligned_alloc(size_t size);
    void aligned_free(void* ptr);

    std::unique_ptr<Kernel> m_kernel;
    std::map<void*, size_t> m_alloc_memory;
    std::map<size_t, std::vector<void*>> m_free_memory;
};

class CPUDevice : public Device {
public:
    CPUDevice(KernelType type, uint32_t nr_thread) : Device() {
        m_thread_pool = make_unique<ThreadPool>(nr_thread);
        m_kernel = make_unique<Kernel>(type, m_thread_pool.get());
    }

    void* allocate(size_t len) override;

    void free_device(void* ptr) override;

    ~CPUDevice();

private:
    std::unique_ptr<ThreadPool> m_thread_pool;
};

class GPUDevice : Device {
public:
    GPUDevice() : Device() { m_kernel = make_unique<Kernel>(KernelType::GPU); }


    ~GPUDevice();

    void* allocate(size_t len) override;

    void free_device(void* ptr) override;

    void* cpu_temp_allocate(size_t len);
    void  copy_data_from_cpu(float* dst, float* src, size_t len);
private:
    std::map<void*, size_t> m_alloc_memory_cpu_temp;
    std::map<size_t, std::vector<void*>> m_free_memory_cpu_temp;
};
}  // namespace inferllm
