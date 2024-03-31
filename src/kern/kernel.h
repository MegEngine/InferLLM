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
#ifndef SRC_KERN_KERNEL_H_
#define SRC_KERN_KERNEL_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "core/thread_pool.h"
#include "kern/kernel_define.h"
#include "utils.h"

namespace llm_learning {
class Kernel {
 public:
  Kernel(KernelType kernel_type, ThreadPool* thread_pool)
      : m_kernel_type(kernel_type), m_thread_pool(thread_pool) {}

  uint32_t nr_thread() const { return m_thread_pool->nr_threads(); }

  //! compute
  template <KernelID Id, typename... Args>
  void operator()(Args... args);

  template <KernelID Id, typename... Args>
  size_t get_workspace(Args... args);
  ThreadPool* m_thread_pool;
  KernelType m_kernel_type;
};
}  // namespace llm_learning

#endif
