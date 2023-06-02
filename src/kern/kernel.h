#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "core/thread_pool.h"
#include "kern/gpu/gpu.h"
#include "kern/kernel_define.h"
#include "kern/naive/naive.h"
#include "kern/optimized/kernel_opt.h"
#include "utils.h"

namespace inferllm {
class Kernel {
public:
    Kernel(KernelType kernel_type) : m_kernel_type(kernel_type) {}
    Kernel(KernelType kernel_type, ThreadPool* thread_pool)
            : m_kernel_type(kernel_type), m_thread_pool(thread_pool) {}

    uint32_t nr_thread() const { return m_thread_pool->nr_threads(); }

    //! compute
    template <KernelID Id, typename... Args>
    void operator()(Args... args) {
        //! parallel to execute tasks
        if (m_kernel_type == KernelType::GPU) {
            gpu::Comp<Id, Args...>::get_all_task(std::forward<Args>(args)...);

        } else {
            TaskSet task_set =
                    opt::Comp<Id, Args...>::get_all_task(std::forward<Args>(args)...);
            for (auto& task : task_set) {
                m_thread_pool->add_task(task.first, task.second);
            }
        }
    };
    template <KernelID Id, typename... Args>
    size_t get_workspace(Args... args) {
        return opt::Space<Id, Args...>::get(std::forward<Args>(args)...);
    }
    ThreadPool* m_thread_pool;
    KernelType m_kernel_type;
};
// namespace inferllm
}  // namespace inferllm