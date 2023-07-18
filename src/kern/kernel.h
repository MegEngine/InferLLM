#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "core/thread_pool.h"
#include "kern/kernel_define.h"
#include "utils.h"

#if INFER_X86
#include "kern/optimized/x86/kernel.h"
#elif INFER_ARM
#include "kern/optimized/arm/kernel.h"
#elif INFER_RV64
#include "kern/optimized/rv64/kernel.h"
#else
#include "kern/naive/naive.h"
#endif

#ifdef ENABLE_GPU
#include "kern/gpu/kernel_gpu.h"
#endif

namespace inferllm {

class Kernel {
public:
    Kernel(KernelType kernel_type) : m_kernel_type(kernel_type) {}
    Kernel(KernelType kernel_type, ThreadPool* thread_pool)
            : m_kernel_type(kernel_type), m_thread_pool(thread_pool) {
#ifdef INFER_RV64
        opt::init();
#endif
    }

    uint32_t nr_thread() const {
        if (m_thread_pool == nullptr)
            return 1;
        return m_thread_pool->nr_threads();
    }

    //! compute
    template <KernelID Id, typename... Args>
    void operator()(Args... args) {
        //! parallel to execute tasks
        if (m_kernel_type == KernelType::GPU) {
#if ENABLE_GPU
            gpu::Comp<Id, Args...>::exec(std::forward<Args>(args)..., m_handle);
#endif

        } else {
            TaskSet task_set =
                    opt::Comp<Id, Args...>::get_all_task(std::forward<Args>(args)...);
            for (auto& task : task_set) {
                m_thread_pool->add_task(task.first, task.second);
            }
        }
    }
    template <KernelID Id, typename... Args>
    size_t get_workspace(Args... args) {
        return opt::Space<Id, Args...>::get(std::forward<Args>(args)...);
    }

    ThreadPool* m_thread_pool = nullptr;
    KernelType m_kernel_type;
#if ENABLE_GPU
    void set_handle(cudaHandle* handle) { m_handle = handle; }
    cudaHandle* m_handle;
#endif
};

}  // namespace inferllm
