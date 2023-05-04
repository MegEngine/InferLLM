#pragma once

#include <functional>
#include <map>

#include "kern/kernel.h"
#include "utils.h"
#include "thread_pool.h"

namespace inferllm {



class Tensor;

//! TODO: this task maybe as class to implement the CPU task and GPU task
using Task = std::function<void(void)>;

class Device {
public:
    Device(KernelType type, uint32_t nr_thread) {
        m_thread_pool = make_unique<ThreadPool>(nr_thread);
        m_kernel = make_unique<Kernel>(type, m_thread_pool.get());
    }
    KernelType type() { return m_kernel->m_kernel_type; };

    virtual ~Device();

    void* allocate(size_t len);

    void free_device(void* ptr);

    //! move the source data from the device
    void move_data_into(void* dst_data, void* src_data, size_t size,
                        const Device& device) {}

    Kernel* kernel() { return m_kernel.get(); }

private:
    std::unique_ptr<Kernel> m_kernel;
    std::unique_ptr<ThreadPool> m_thread_pool;
    std::map<size_t, std::vector<void*>> m_free_memory;
    std::map<void*, size_t> m_alloc_memory;

    void* aligned_alloc(size_t size);
    void aligned_free(void* ptr);
};

}  // namespace inferllm

