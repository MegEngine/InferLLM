#include "thread_pool.h"

using namespace inferllm;

ThreadPool::ThreadPool(uint32_t threads_num)
        : m_nr_threads(threads_num), m_stop{false}, m_active{false} {
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
                while (!m_stop) {
                    while (m_active) {
                        //! if the thread should work
                        if (m_workers[i]->work_flag.load(std::memory_order_acquire)) {
                            m_task(TaskId{
                                    i * m_task_per_thread,
                                    std::min((i + 1) * m_task_per_thread, m_nr_task),
                                    i});
                            //! Flag worker is finished
                            m_workers[i]->work_flag.store(
                                    false, std::memory_order_release);
                        }
                        //! Wait next task coming
                        for (int it = 0; it < WORKER_ACTIVE_WAIT; it++) {
                            if (m_workers[i]->work_flag.load(
                                        std::memory_order_acquire)) {
                                break;
                            }
                            if (it < ACTIVE_WAIT_PAUSE_LIMIT || (it & 1)) {
                                INFER_PAUSE(16);  // Spin lock's CPU-level yield
                            } else {
                                // Spin lock's OS-level yield
                                std::this_thread::yield();
                            }
                        }
                    }
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        if (!m_stop && !m_active) {
                            m_cv.wait(lock, [this] { return m_stop || m_active; });
                        }
                    }
                }
            }));
        }
    }
}
void ThreadPool::add_task(const MultiThreadingTask& task, uint32_t nr_task) {
    //! If only one thread or one task, execute directly
    if (m_nr_threads == 1 || nr_task == 1) {
        task({0, nr_task, m_nr_threads - 1});
        return;
    } else {
        active();
        INFER_ASSERT(m_active, "thread pool is not actived.");
        m_nr_task = nr_task;
        //! Set the task number, task iter and task
        m_task_per_thread = (nr_task + m_nr_threads - 1) / m_nr_threads;
        m_task = std::move(task);
        for (uint32_t i = 0; i < m_nr_threads - 1; i++) {
            m_workers[i]->work_flag.store(true, std::memory_order_release);
        }
        //! Main thread working
        uint32_t start = (m_nr_threads - 1) * m_task_per_thread;
        // printf("main threads start\n");
        m_task({start, nr_task, m_nr_threads - 1});
        //! make sure all threads done
        sync();
    }
}

inline void ThreadPool::sync() {
    bool no_finished = false;
    uint32_t no_finished_id = 0;
    do {
        no_finished = false;
        for (uint32_t i = no_finished_id; i < m_nr_threads - 1; ++i) {
            if (m_workers[i]->work_flag.load(std::memory_order_acquire)) {
                no_finished = true;
                no_finished_id = i;
                break;
            }
        }
        if (no_finished) {
            for (int it = 0; it < MAIN_THREAD_ACTIVE_WAIT; it++) {
                if (!m_workers[no_finished_id]->work_flag.load(
                            std::memory_order_acquire)) {
                    break;
                }
                if ((it < ACTIVE_WAIT_PAUSE_LIMIT || (it & 1))) {
                    INFER_PAUSE(16);
                } else {
                    std::this_thread::yield();
                }
            }
        }
    } while (no_finished);
}
inline void ThreadPool::active() {
    if (!m_active) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_active = true;
        m_cv.notify_all();
    }
}
void ThreadPool::deactive() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_active = false;
}
ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_stop = true;
        m_active = false;
        m_cv.notify_all();
    }
    for (auto& worker : m_workers) {
        delete worker;
    }
}
