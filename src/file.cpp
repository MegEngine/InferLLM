#include "file.h"
#include "utils.h"
#include "string.h"

using namespace inferllm;

InputFile::InputFile(const std::string& path, bool enable_mmap)
        : m_enable_mmap{enable_mmap} {
    m_file = fopen(path.c_str(), "rb");
    INFER_ASSERT(m_file, "Failed to open model file.");
    m_fd = fileno(m_file);
    fseek(m_file, 0, SEEK_END);
    m_size = ftell(m_file);
    rewind();
    if (m_enable_mmap) {
        int flags = MAP_SHARED;
        m_mmap_addr = mmap(NULL, m_size, PROT_READ, flags, m_fd, 0);
        INFER_ASSERT(m_mmap_addr != MAP_FAILED, "mmap failed.");
        madvise(m_mmap_addr, m_size, MADV_WILLNEED);
    }
}

void* InputFile::get_mmap_data(size_t len, size_t offset) {
    INFER_ASSERT(offset < m_size, "offset error when get mmap data.");
    return static_cast<void*>(static_cast<int8_t*>(m_mmap_addr) + offset);
}

std::uint32_t InputFile::read_u32() {
    std::uint32_t ret;
    read_raw(&ret, sizeof(ret));
    return ret;
}

std::string InputFile::read_string(std::uint32_t len) {
    std::vector<char> chars(len);
    read_raw(chars.data(), len);
    return std::string(chars.data(), len);
}

void InputFile::skip(int64_t bytes) {
    auto err = fseek(m_file, bytes, SEEK_CUR);
    INFER_ASSERT(!err, "skip file error");
}

void InputFile::seek(size_t offset, FilePos pos) {
    auto err = fseek(m_file, offset, (int)pos);
    INFER_ASSERT(!err, "skip file error");
}

void InputFile::read_raw(void* dst, size_t size) {
    if (size == 0) {
        return;
    }
    auto nr = fread(dst, 1, size, m_file);
    INFER_ASSERT(nr == size, "read file error");
}

void InputFile::read_data(void* dst, size_t size, size_t offset) {
    fseek(m_file, offset, SEEK_SET);
    read_raw(dst, size);
}