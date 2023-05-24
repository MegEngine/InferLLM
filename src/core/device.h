#pragma once

#include <functional>
#include <map>

#include "kern/kernel.h"
#include "thread_pool.h"
#include "utils.h"

#if ENABLE_GPU
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
    Device() = default;

    virtual void* allocate(size_t len) = 0;
    virtual void free_device(void* ptr) = 0;
    virtual ~Device() = default;

    Kernel* kernel() { return m_kernel.get(); }

    KernelType type() { return m_kernel->m_kernel_type; };

    virtual void* aligned_alloc(size_t size);

    virtual void aligned_free(void* ptr);

    virtual void deactive() {}

    virtual void host2device_copy(void* device, const void* host, size_t size) = 0;

    virtual void device2host_copy(void* host, const void* device, size_t size) = 0;

    virtual void device2device_copy(void* dst, const void* src, size_t size) = 0;

protected:
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

    void deactive() override { m_thread_pool->deactive(); }

    void host2device_copy(void* device, const void* host, size_t size) override {
        memcpy(device, host, size);
    }

    void device2host_copy(void* host, const void* device, size_t size) override {
        memcpy(host, device, size);
    }

    void device2device_copy(void* dst, const void* src, size_t size) override {
        memcpy(dst, src, size);
    }

    ~CPUDevice();

private:
    std::unique_ptr<ThreadPool> m_thread_pool;
};

#if ENABLE_GPU
class GPUDevice : public Device {
public:
    GPUDevice(int device) : Device() {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaStreamCreate(&m_stream));
        m_kernel = make_unique<Kernel>(KernelType::GPU);
        m_kernel->set_stream(m_stream);
    }

    ~GPUDevice();

    void* allocate(size_t len) override;

    void free_device(void* ptr) override;

    void host2device_copy(void* device, const void* host, size_t size) override;

    void device2host_copy(void* host, const void* device, size_t size) override;

    void device2device_copy(void* dst, const void* src, size_t size) override;

private:
    cudaStream_t m_stream{nullptr};
};

#endif
}  // namespace inferllm
