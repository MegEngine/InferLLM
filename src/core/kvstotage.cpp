#include"kvstorage.h"

using namespace inferllm;

void KvStorage::set_shared_memory(void* data, size_t size) {
    Tensor::set_shared_memory(data, size);
    m_curr_data = static_cast<char*>(ptr()) +
                  static_cast<size_t>(
                          (stride()[0] * m_store_id * dtype_in_byte(dtype())));
}

TensorState KvStorage::prepare_data() {
    Tensor::prepare_data();
    m_curr_data = static_cast<char*>(ptr()) +
                  static_cast<size_t>(
                          (stride()[0] * m_store_id * dtype_in_byte(dtype())));
    return TensorState::Own;
}
