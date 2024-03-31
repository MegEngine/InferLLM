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
 
#include "kvstorage.h"

using namespace llm_learning;

void KvStorage::set_shared_memory(void* data, size_t size) {
  Tensor::set_shared_memory(data, size);
  m_curr_data = static_cast<char*>(ptr()) +
                static_cast<size_t>((stride()[0] * m_store_id * dtype_in_byte(dtype())));
}

TensorState KvStorage::prepare_data() {
  Tensor::prepare_data();
  m_curr_data = static_cast<char*>(ptr()) +
                static_cast<size_t>((stride()[0] * m_store_id * dtype_in_byte(dtype())));
  return TensorState::Own;
}
