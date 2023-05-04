#pragma once

#include "tensor.h"

namespace inferllm {

class KvStorage : public Tensor {
public:
    KvStorage(std::vector<size_t> shape, DType dtype, Device* device)
            : Tensor(shape, dtype, device) {
        m_store_id = 0;
        m_total_id = shape[0];
        size_t len = length_in_byte();
        auto data = device->allocate(len);
        set_shared_memory(data, len);
    }

    ~KvStorage() {
        auto data = ptr();
        device()->free_device(data);
    }
    void* get_current_data() {
        INFER_ASSERT(
                is_own(),
                "The Kvstorage is not ready, please call prepare_data ahead.");
        return m_curr_data;
    }

    void set_shared_memory(void* data, size_t length = 0) override;

    TensorState prepare_data() override;

    size_t add_id(uint32_t id) {
        INFER_ASSERT(id + m_store_id < m_total_id, "KvStorage add id error!");
        m_store_id += id;
        m_curr_data = static_cast<char*>(ptr()) +
                      static_cast<size_t>((stride()[0] * m_store_id *
                                           dtype_in_byte(dtype())));
        return m_store_id;
    }

    size_t current_index() const { return m_store_id; }

    // reset the current index to 0, and set the current data to the first
    void reset_id() {
        m_store_id = 0;
        m_curr_data = ptr();
    }

private:
    size_t m_store_id;
    size_t m_total_id;
    void* m_curr_data;
};
}  // namespace inferllm
