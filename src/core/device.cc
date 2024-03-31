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

/*!
 * \file device.cc
 * \brief The implement of device struct.
 */

#include "device.h"

#include "tensor.h"

#ifndef __APPLE__
#include <malloc.h>
#endif

#define ALIGN_SIZE (32)

using namespace llm_learning;

void* Device::aligned_alloc(size_t size) {
#ifdef WIN32
  return _aligned_malloc(size, ALIGN_SIZE);
#elif defined(__ANDROID__) || defined(ANDROID)
  return memalign(ALIGN_SIZE, size);
#else
  void* ptr = nullptr;
  auto err = posix_memalign(&ptr, ALIGN_SIZE, size);
  INFER_ASSERT(!err, "failed to malloc.");
  return ptr;
#endif
}

void Device::aligned_free(void* ptr) {
#ifdef WIN32
  _aligned_free(ptr);
#else
  ::free(ptr);
#endif
}

size_t nr_allocate = 0;

void* Device::allocate(size_t len) {
  auto it = m_free_memory.lower_bound(len);
  void* ptr = nullptr;
  if (it != m_free_memory.end() && it->second.size() > 0) {
    ptr = it->second.back();
    it->second.pop_back();
    if (it->second.size() < 1) {
      m_free_memory.erase(it);
    }
  } else {
    ptr = aligned_alloc(len);
    nr_allocate += len;
    m_alloc_memory[ptr] = len;
  }
  return ptr;
}

void Device::free_device(void* ptr) {
  INFER_ASSERT(m_alloc_memory.count(ptr) == 1, "memory is not allocated by the DeviceCPU.");
  size_t len = m_alloc_memory[ptr];
  m_free_memory[len].push_back(ptr);
}

Device::~Device() {
  for (auto it : m_free_memory) {
    for (auto ptr : it.second) {
      INFER_ASSERT(m_alloc_memory.count(ptr) == 1, "memory is not allocated by the DeviceCPU.");
      aligned_free(ptr);
    }
  }
  INFER_LOG("total allocate memory is %zu\n", nr_allocate);
}
