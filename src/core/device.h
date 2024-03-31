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

#ifndef SRC_CORE_DEVICE_H_
#define SRC_CORE_DEVICE_H_

#include <functional>
#include <map>
#include <memory>

#include "kern/kernel.h"
#include "thread_pool.h"
#include "utils.h"

namespace llm_learning {

class Tensor;

//! TODO: this task maybe as class to implement the CPU task and GPU task
using Task = std::function<void(void)>;

class Device {
 public:
  Device(KernelType type, uint32_t nr_thread) {
    m_thread_pool = std::make_unique<ThreadPool>(nr_thread);
    m_kernel = std::make_unique<Kernel>(type, m_thread_pool.get());
  }
  KernelType type() { return m_kernel->m_kernel_type; };

  virtual ~Device();

  void* allocate(size_t len);

  void free_device(void* ptr);

  //! move the source data from the device
  void move_data_into(void* dst_data, void* src_data, size_t size, const Device& device) {}

  Kernel* kernel() { return m_kernel.get(); }

 private:
  std::unique_ptr<Kernel> m_kernel;
  std::unique_ptr<ThreadPool> m_thread_pool;
  std::map<size_t, std::vector<void*>> m_free_memory;
  std::map<void*, size_t> m_alloc_memory;

  void* aligned_alloc(size_t size);
  void aligned_free(void* ptr);
};

}  // namespace llm_learning

#endif  // SRC_CORE_DEVICE_H_