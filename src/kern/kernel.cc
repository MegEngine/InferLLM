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
 
#include "kernel.h"

#include "kern/optimized/kernel_opt.h"

using namespace llm_learning;

//! compute
template <KernelID Id, typename... Args>
void Kernel::operator()(Args... args) {
  TaskSet task_set = opt::Comp<Id, Args...>::get_all_task(std::forward<Args>(args)...);
  //! parallel to execute tasks
  for (auto& task : task_set) {
    m_thread_pool->add_task(task.first, task.second);
  }
}

template <KernelID Id, typename... Args>
size_t Kernel::get_workspace(Args... args) {
  return opt::Space<Id, Args...>::get(std::forward<Args>(args)...);
}