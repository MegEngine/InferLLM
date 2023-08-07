#include "thread_pool.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include "Tracy.hpp"

using namespace inferllm;

inline void __thread_relax() noexcept {
#if defined __i386__ || defined __x86_64__
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

ThreadPool::ThreadPool(uint32_t threads_num) : m_nr_threads(threads_num) {
    if (threads_num < 1) {
        m_nr_threads = 1;
    }
    if (m_nr_threads > 1) {
        auto system_cpu_count = std::thread::hardware_concurrency();
        if (m_nr_threads > system_cpu_count) {
            INFER_LOG(
                    "The number of threads is bigger than number of "
                    "physical cpu cores, got: %d core_number: %d",
                    system_cpu_count, nr_threads());
        }
        for (uint32_t i = 0; i < m_nr_threads - 1; i++) {
            m_workers.push_back(new Worker([this, i]() {
                {
                    std::unique_lock<std::mutex> lk(m_mtx);
                    m_cv.wait(lk);
                }
                auto& w = m_workers[i]->work_flag;
                while (true) {
                    for (int j = 0;; j++) {
                        worker_status_t s = w.load(std::memory_order_consume);
                        if (s == WRK_START) {
                            break;
                        } else if (s == WRK_STOP)
                            return;
                        if (s == WRK_IDLE) {
                            std::unique_lock<std::mutex> lk(m_mtx);
                            m_cv.wait(lk);
                        }
                        if (j & 0x1F)
                            __thread_relax();
                        else if (j & 0xFF)
                            std::this_thread::yield();
                    }

                    //! if the thread should work
                    // printf("thread %d work form %d to %d\n", i,
                    //        i * m_task_per_thread,
                    //        (i + 1) * m_task_per_thread);
                    m_task(TaskId{
                            (i+1) * m_task_per_thread,
                            std::min((i + 2) * m_task_per_thread, m_nr_task), i});
                    // printf("thread %d finished\n", i);
                    //! Flag worker is finished
                    w.store(WRK_BUSY_WAIT, std::memory_order::memory_order_release);
                }
            }));
        }
        m_cv.notify_all();
    }
}
void ThreadPool::add_task(const MultiThreadingTask& task, uint32_t nr_task) {
    ZoneScopedNS("add_task", 4);
    //! If only one thread or one task, execute directly
    if (m_nr_threads == 1 || nr_task == 1) {
        task({0, nr_task, m_nr_threads - 1});
        return;
    } else {
        m_nr_task = nr_task;
        //! Set the task number, task iter and task
        m_task_per_thread = (nr_task + m_nr_threads - 1) / m_nr_threads;
        m_task = std::move(task);
        for (auto w : m_workers)
            w->work_flag.store(WRK_START, std::memory_order::memory_order_release);
        // printf("main threads start\n");
        m_task({0, m_task_per_thread, m_nr_threads - 1});
        //! make sure all threads done
        int m = 0;
        for (int j = 0;; j++) {
            bool s = true;
            for (int n = m; n < m_workers.size(); n++) {
                if (m_workers[n]->work_flag.load(
                            std::memory_order::memory_order_acquire) == WRK_START) {
                    m = n;
                    s = false;
                    break;
                }
            }
            if (s)
                break;
            if (j & 0x7)
                std::this_thread::yield();
        }
    }
}

void ThreadPool::active() {
    for (auto w : m_workers)
        w->work_flag.store(WRK_BUSY_WAIT, std::memory_order::memory_order_release);
    m_cv.notify_all();
}
void ThreadPool::deactive() {
    for (auto w : m_workers)
        w->work_flag.store(WRK_IDLE, std::memory_order::memory_order_release);
}
ThreadPool::~ThreadPool() {
    for (auto w : m_workers)
        w->work_flag.store(WRK_STOP, std::memory_order::memory_order_release);
    for (auto& worker : m_workers) {
        delete worker;
    }
}
