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

namespace inferllm {

typedef enum {
    WRK_IDLE = 0,   // use cv to lock
    WRK_STOP,       // exit
    WRK_BUSY_WAIT,  // busy loop
    WRK_START,      // real start
} worker_status_t;

/**
 * \brief Worker and related flag
 */
struct Worker {
public:
    Worker(std::function<void()>&& run) : thread{run} {}
    ~Worker() { thread.join(); }
    //! Worker thread
    std::thread thread;
    std::atomic<worker_status_t> work_flag{WRK_IDLE};
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
    void active();
    //! all the threads go to sleep which will reduce CPU occupation
    void deactive();
    ~ThreadPool();

    uint32_t nr_threads() const { return m_nr_threads; }

private:
    uint32_t m_nr_threads = 1;
    //! All the sub task number
    uint32_t m_nr_task = 0;
    uint32_t m_task_per_thread = 0;
    //! The executable funcition pointer
    MultiThreadingTask m_task;

    std::vector<Worker*> m_workers;
    std::condition_variable m_cv;
    std::mutex m_mtx;
};

}  // namespace inferllm
