/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef SRC_CORE_THREAD_POOL_H_
#define SRC_CORE_THREAD_POOL_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "kern/kernel_define.h"
#include "utils.h"

namespace llm_learning {

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

}  // namespace llm_learning

#endif