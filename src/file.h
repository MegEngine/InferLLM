#pragma once

#include <cstdio>
#include <string>
#include <cstdint>

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

namespace inferllm {

enum class FilePos {
    Begin = 0,
    Current = 1,
    End = 2,
};

class InputFile {
    FILE* m_file = nullptr;
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

    void seek(size_t offset, FilePos pos = FilePos::Begin);

    void read_raw(void* dst, size_t size);

    void read_data(void* dst, size_t size, size_t offset);

    size_t tell() { return std::ftell(m_file); }

    void* get_mmap_data(size_t len, size_t offset);

    std::uint32_t read_u32();

    std::string read_string(std::uint32_t len);
};

}  // namespace inferllm
