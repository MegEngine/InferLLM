#include "kvstorage.h"

using namespace inferllm;

class KvStorageConfig {
public:
    constexpr static uint32_t START_KV_INDEX = 100;
    constexpr static uint32_t KV_STEP = 100;
    static std::shared_ptr<KvStorageConfig> instance;
    static std::shared_ptr<KvStorageConfig> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<KvStorageConfig>();
        }
        return instance;
    }
    uint32_t increase_count() {
        m_kv_count++;
        return m_kv_count;
    }

    uint32_t get_start_index() { return START_KV_INDEX + m_kv_count * 2; }

    uint32_t get_count() { return m_kv_count; }

private:
    uint32_t m_kv_count = 0;
};

std::shared_ptr<KvStorageConfig> KvStorageConfig::instance = nullptr;

KvStorage::KvStorage(std::vector<size_t> shape, DType dtype, Device* device)
        : Tensor(device, "kvstorage") {
    m_store_id = 0;
    m_total_id = shape[0];
    m_kv_id = KvStorageConfig::get_instance()->increase_count();
    m_curr_id = KvStorageConfig::get_instance()->get_start_index();
    //! only allocate the memory of length m_curr_id * embd
    shape[0] = m_curr_id;
    set_shape(shape, dtype);
    size_t len = length_in_byte();
    //! no need use memory pool
    auto data = device->aligned_alloc(len);
    set_shared_memory(data, len);
}

void KvStorage::set_shared_memory(void* data, size_t size) {
    Tensor::set_shared_memory(data, size);
    m_curr_data =
            static_cast<char*>(ptr()) +
            static_cast<size_t>((stride()[0] * m_store_id * dtype_in_byte(dtype())));
}

TensorState KvStorage::prepare_data_with_length(uint32_t len) {
    Tensor::prepare_data();
    //! if memory is not enough, allocate a new memory and copy the data to the new
    if (m_store_id + len >= m_curr_id) {
        auto shape = this->shape();
        shape[0] = m_curr_id + KvStorageConfig::KV_STEP;
        size_t old_len = length_in_byte();
        void* old_ptr = ptr();

        set_shape(shape, dtype());
        size_t len = length_in_byte();
        auto data = device()->aligned_alloc(len);
        device()->device2device_copy(data, old_ptr, old_len);

        device()->aligned_free(old_ptr);

        set_shared_memory(data, len);
        m_curr_id += KvStorageConfig::KV_STEP;
    }
    m_curr_data =
            static_cast<char*>(ptr()) +
            static_cast<size_t>((stride()[0] * m_store_id * dtype_in_byte(dtype())));
    return TensorState::Own;
}
