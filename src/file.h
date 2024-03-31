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
 
#ifndef SRC_FILE_H_
#define SRC_FILE_H_

#include <cstdint>
#include <cstdio>
#include <string>

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif  // _POSIX_MAPPED_FILES
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif  //_POSIX_MEMLOCK_RANGE
#endif  // __has_include(<unistd.h>)
#endif  // __has_include

namespace llm_learning {

class InputFile {
  FILE* m_file;
  int m_fd;
  size_t m_size;
  bool m_enable_mmap = false;
  void* m_mmap_addr = nullptr;

 public:
  InputFile(const std::string& path, bool enable_mmap = false);

  ~InputFile() {
    if (m_file) {
      fclose(m_file);
    }
    if (m_enable_mmap) {
      munmap(m_mmap_addr, m_size);
    }
  }

  bool enable_mmap() { return m_enable_mmap; }

  bool eof() { return tell() == m_size; }

  void rewind() { std::rewind(m_file); }

  void skip(int64_t bytes);

  void read_raw(void* dst, size_t size);

  void read_data(void* dst, size_t size, size_t offset);

  size_t tell() { return std::ftell(m_file); }

  void* get_mmap_data(size_t len, size_t offset);

  std::uint32_t read_u32();

  std::string read_string(std::uint32_t len);
};

}  // namespace llm_learning

#endif