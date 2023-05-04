
#include "device.h"
#include "tensor.h"

#ifndef __APPLE__
#include <malloc.h>
#endif

#define ALIGN_SIZE (32)

using namespace inferllm;

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
    INFER_ASSERT(m_alloc_memory.count(ptr) == 1,
                 "memory is not allocated by the DeviceCPU.");
    size_t len = m_alloc_memory[ptr];
    m_free_memory[len].push_back(ptr);
}

Device::~Device() {
    for (auto it : m_free_memory) {
        for (auto ptr : it.second) {
            INFER_ASSERT(m_alloc_memory.count(ptr) == 1,
                         "memory is not allocated by the DeviceCPU.");
            aligned_free(ptr);
        }
    }
    INFER_LOG("total allocate memory is %zu\n", nr_allocate);
}
