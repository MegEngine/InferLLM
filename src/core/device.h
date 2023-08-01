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

    virtual void* allocate_host(size_t len) = 0;

    virtual void free_device(void* ptr) = 0;

    virtual void free_host(void* ptr) = 0;

    virtual ~Device() = default;

    Kernel* kernel() { return m_kernel.get(); }

    KernelType type() { return m_kernel->m_kernel_type; };

    virtual void* aligned_alloc(size_t size);

    virtual void aligned_free(void* ptr);

    virtual void deactive() {}

    virtual void host2device_copy(
            void* device, const void* host, size_t size, bool async = false) = 0;

    virtual void device2host_copy(
            void* host, const void* device, size_t size, bool async = false) = 0;

    virtual void device2device_copy(
            void* dst, const void* src, size_t size, bool async = false) = 0;

    virtual void sync() = 0;

    //! whether the device support unified memory, in  unified memory, the host memory
    //! and device memory is the same, CPU is the unified memory, GPU is the not unified
    //! memory
    virtual bool unified_memory() { return true; }

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
    void* allocate_host(size_t len) override;

    void free_device(void* ptr) override;
    void free_host(void* ptr) override;

    void deactive() override { m_thread_pool->deactive(); }

    void host2device_copy(
            void* device, const void* host, size_t size, bool async = false) override {
        memcpy(device, host, size);
    }

    void device2host_copy(
            void* host, const void* device, size_t size, bool async = false) override {
        memcpy(host, device, size);
    }

    void device2device_copy(
            void* dst, const void* src, size_t size, bool async = false) override {
        memcpy(dst, src, size);
    }

    void sync() override {}

    ~CPUDevice();

private:
    std::unique_ptr<ThreadPool> m_thread_pool;
};

#if ENABLE_GPU
class GPUDevice : public Device {
public:
    GPUDevice(int device) : Device() {
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaStreamCreate(&(m_handle.stream)));
        CUBLAS_CHECK(cublasCreate(&(m_handle.cublas_handle)));
        m_kernel = make_unique<Kernel>(KernelType::GPU);
        m_kernel->set_handle(&m_handle);
    }

    ~GPUDevice();

    void* allocate(size_t len) override;
    void* allocate_host(size_t len) override;

    void free_device(void* ptr) override;
    void free_host(void* ptr) override;

    void* aligned_alloc(size_t size) override {
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        return ptr;
    }

    void aligned_free(void* ptr) override { CUDA_CHECK(cudaFree(ptr)); }

    void host2device_copy(
            void* device, const void* host, size_t size, bool async = false) override;

    void device2host_copy(
            void* host, const void* device, size_t size, bool async = false) override;

    void device2device_copy(
            void* dst, const void* src, size_t size, bool async = false) override;

    void sync() override { CUDA_CHECK(cudaStreamSynchronize(m_handle.stream)); }

    bool unified_memory() override { return false; }

private:
    cudaHandle m_handle;
};

#endif
}  // namespace inferllm
