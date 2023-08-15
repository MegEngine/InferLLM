#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "kern/kernel_define.h"
#include "utils.h"

// clang-format off
#ifndef INFER_PAUSE
# if defined __GNUC__ && (defined __i386__ || defined __x86_64__)
#   if !defined(__SSE2__)
      static inline void non_sse_mm_pause() { __asm__ __volatile__ ("rep; nop"); }
#     define _mm_pause non_sse_mm_pause
#   else
#       include <immintrin.h>
#   endif
#   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { _mm_pause(); } } while (0)
# elif defined __GNUC__ && defined __aarch64__
#   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("yield" ::: "memory"); } } while (0)
# elif defined __GNUC__ && defined __arm__
#   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("" ::: "memory"); } } while (0)
# elif defined __GNUC__ && defined __riscv
// PAUSE HINT is not part of RISC-V ISA yet, but is under discussion now. For details see:
// https://github.com/riscv/riscv-isa-manual/pull/398
// https://github.com/riscv/riscv-isa-manual/issues/43
// #   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("pause"); } } while (0)
#   define INFER_PAUSE(v) do { for (int __delay = (v); __delay > 0; --__delay) { asm volatile("nop"); } } while (0)
# else
#   warning "Can't detect 'pause' (CPU-yield) instruction on the target platform. Specify INFER_PAUSE() definition via compiler flags."
#   define INFER_PAUSE(...) do { /* no-op: works, but not effective */ } while (0)
# endif
#endif // MTDA_PAUSE
// clang-format on

namespace inferllm {

/**
 * \brief Worker and related flag
 */
struct Worker {
public:
    Worker(std::function<void()>&& run) : thread{run} {}
    ~Worker() { thread.join(); }
    //! Worker thread
    std::thread thread;
    //! Indicate whether the Worker thread need run
    std::atomic<bool> work_flag{false};
};

/**
 * \brief ThreadPool execute the task in multi-threads(nr_threads>1) mode , it
 * will fallback to single-thread mode if nr_thread is 1.
 */
class ThreadPool {
public:
    //! Create thread-pool nr_threads thread_pool
    ThreadPool(uint32_t nr_threads);
    //! The main thread set the task, parallelism and worker flag to
    //! notify other thread.
    void add_task(const MultiThreadingTask& task, uint32_t nr_task);

    inline void sync();
    //! wake up all the threads from cv.wait(), when the thread pool is not
    //! active, all the threads will go to sleep.
    inline void active();
    //! all the threads go to sleep which will reduce CPU occupation
    void deactive();
    ~ThreadPool();

    uint32_t nr_threads() const { return m_nr_threads; }

    //! The number of iterations < main thread yeild resource>
    static constexpr int MAIN_THREAD_ACTIVE_WAIT = 10000;
    //! The number of iterations < worker thread yeild resource>
    static constexpr int WORKER_ACTIVE_WAIT = 2000;
    //! The number of iterations <pause>
    static constexpr int ACTIVE_WAIT_PAUSE_LIMIT = 16;

private:
    uint32_t m_nr_threads = 1;
    //! All the sub task number
    uint32_t m_nr_task = 0;
    uint32_t m_task_per_thread = 0;
    std::atomic_bool m_stop{false};
    std::atomic_bool m_active{false};
    //! The executable funcition pointer
    MultiThreadingTask m_task;

    std::vector<Worker*> m_workers;
    //! The cv and mutex for threading activity
    std::condition_variable m_cv;
    std::mutex m_mutex;
};

}  // namespace inferllm