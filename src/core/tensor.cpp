#include "tensor.h"
#include "memory.h"
#include "utils.h"


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
        if (m_file) {
            //! if the tensor data is from file, we can map the memory from file
            //! or read the data from file
            if (m_file->enable_mmap()) {
                m_data = m_file->get_mmap_data(length, m_file_offset);
            } else if (m_data == nullptr) {
                m_data = m_device->allocate(length);
                m_file->read_data(m_data, length, m_file_offset);
            }
            //! if the tensor data is from device
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

void Tensor::set_shared_memory(void* data, size_t size) {
    INFER_ASSERT(data == nullptr || size >= length_in_byte(),
                 "the memory set to tensor is not enough");
    m_state = TensorState::Own;
    m_data = data;
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
