#include "tensor.h"
#include "../kern/kernel_define.h"
#include "memory.h"
#include "utils.h"
#include "op.h"

using namespace inferllm;

float inferllm::dtype_in_byte(DType dtype) {
    switch (dtype) {
        case DType::Float32:
        case DType::Int32:
            return 4;
        case DType::Float16:
        case DType::Int16:
            return 2;
        case DType::Float8:
        case DType::Uint8:
            return 1;
        case DType::Int8:
            return sizeof(BlockQ80);
        case DType::Int4:
            //! QK number int4 as a block, and share a float scale
            return sizeof(BlockQ40);
        default:
            INFER_ASSERT(0, "No support data type.");
    }
}

uint32_t inferllm::dtype_block_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
        case DType::Int32:
        case DType::Float16:
        case DType::Int16:
        case DType::Float8:
        case DType::Uint8:
            return 1;
        case DType::Int8:
            return QK80;
        case DType::Int4:
            return QK40;
        default:
            INFER_ASSERT(0, "No support data type.");
    }
}

TensorState Tensor::prepare_data() {
    size_t length = length_in_byte();
    if (!m_data && m_state == TensorState::OutSide) {
        //! if m_file is not nullptr, the tensor is weights and should be read or map
        //! from file
        if (m_file) {
            read_data_from_file();
            //! the data is the input/output tensor of Operator, and should be allocate
            //! from memory pool
        } else {
            m_data = m_device->allocate(length);
        }
    }
    m_state = TensorState::Own;
    return m_state;
}

TensorState Tensor::recall_data() {
    if (m_shared) {
        return m_state;
    }
    //! if the tensor data is from allocate by itself, we need free the memory
    if (!m_file && m_data != nullptr && m_state == TensorState::Own) {
        m_device->free_device(m_data);
        m_data = nullptr;
    }
    m_state = TensorState::OutSide;
    return m_state;
}

size_t Tensor::read_data_from_file() {
    size_t length = length_in_byte();
    if (m_file->enable_mmap()) {
        //! no unified memory, we need read data to host memory and copy to device
        if (!m_device->unified_memory()) {
            auto temp_ptr = m_file->get_mmap_data(length, m_file_offset);
            m_data = m_device->allocate(length);
            m_device->host2device_copy(m_data, temp_ptr, length);
        } else {
            m_data = m_file->get_mmap_data(length, m_file_offset);
        }
    } else if (m_data == nullptr) {
        //! no unified memory, we need read data to host memory and copy to device
        if (!m_device->unified_memory()) {
            m_data = m_device->allocate(length);
            auto host_ptr = m_device->allocate_host(length);
            auto opr = this->owner_op();
            if (opr->need_preprocess_weight(this)) {
                auto host_ptr2 = m_device->allocate_host(length);
                m_file->read_data(host_ptr2, length, m_file_offset);
                auto shape = opr->preprocess_weight(this, host_ptr2, host_ptr);
                set_shape(shape);
                m_device->free_host(host_ptr2);
            } else {
                m_file->read_data(host_ptr, length, m_file_offset);
            }
            m_device->host2device_copy(m_data, host_ptr, length);
            m_device->free_host(host_ptr);
        } else {
            m_data = m_device->allocate(length);
            auto opr = this->owner_op();
            if (opr->need_preprocess_weight(this)) {
                auto host_data = m_device->allocate_host(length);
                m_file->read_data(host_data, length, m_file_offset);
                auto shape = opr->preprocess_weight(this, host_data, m_data);
                set_shape(shape);
                m_device->free_host(host_data);
            } else {
                m_file->read_data(m_data, length, m_file_offset);
            }
        }
    }
    return length;
}

void Tensor::set_shared_memory(void* data, size_t size) {
    INFER_ASSERT(
            data == nullptr || size >= length_in_byte(),
            "the memory set to tensor is not enough");
    m_data = data;
    m_state = TensorState::Own;
    m_shared = true;
}

Tensor::~Tensor() {
    if (m_state == TensorState::Own) {
        recall_data();
    }
    //! the data read from file by m_file->read_data
    if (m_file && !m_file->enable_mmap() && m_data) {
        m_device->free_device(m_data);
    }
}
