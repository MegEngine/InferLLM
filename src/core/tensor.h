#pragma once

#include "device.h"
#include "utils.h"
#include "file.h"
namespace inferllm {

enum class DType {
    Float32 = 0,
    Float16 = 1,
    Float8 = 2,
    Int32 = 3,
    Int16 = 4,
    Int8 = 5,
    Uint8 = 6,
    Int4 = 7,
    Uint4 = 8,
    Int2 = 9,
};

float dtype_in_byte(DType dtype);

//! the data arrangement
uint32_t dtype_block_size(DType dtype);

enum class TensorState {
    Own = 0,
    OutSide = 1,
};

class OpBase;

//! the tensor memory is from three ways:
//! 1. the tensor is own the memory, allocate by itself
//! 2. the tensor memory is shared from outside, such as the input tensor,
//! output tensor
//! 3. the tensor memory is map from file, such as the weight tensor
class Tensor {
public:
    Tensor(Device* device, std::string name)
            : m_device(device), m_name(name) {
        m_state = TensorState::OutSide;
    }

    Tensor(std::vector<size_t> shape, DType dtype, Device* device) {
        m_device = device;
        set_shape(shape);
        set_dtype(dtype);
        m_state = TensorState::OutSide;
    }

    ~Tensor();

    std::vector<size_t> shape() { return m_shape; }

    void set_shape(std::vector<size_t> shape, DType dtype) {
        set_shape(shape);
        set_dtype(dtype);
    }

    void set_shape(std::vector<size_t> shape) {
        m_dims = shape.size();
        m_shape = shape;
        //! init the tensor as continue tensor
        m_stride.resize(m_dims);
        m_stride[m_dims - 1] = 1;
        for (uint32_t i = 1; i < m_dims; i++) {
            m_stride[m_dims - 1 - i] =
                    m_stride[m_dims - i] * m_shape[m_dims - i];
        }
        m_length = m_shape[0] * m_stride[0];
    }

    void set_dtype(DType dtype) { m_dtype = dtype; }
    DType dtype() { return m_dtype; }

    std::vector<size_t> stride() { return m_stride; }

    OpBase* owner_op() { return m_owner_op; }
    void set_owner_op(OpBase* owner_op) { m_owner_op = owner_op; }

    std::string name() { return m_name; }
    void set_name(const std::string& name) { m_name = name; }

    uint32_t dims() { return m_dims; }

    bool is_own() const { return m_state == TensorState::Own; }

    size_t length() { return m_length; }

    Device* device() { return m_device; }

    size_t length_in_byte() {
        //! TODO: assert the length is int
        //! uint4 and int4 data arrangement: 32 data as a blcok and share the
        //! same scale and zero
        return m_length * dtype_in_byte(m_dtype) / dtype_block_size(m_dtype);
    };

    void* ptr() {
        INFER_ASSERT(is_own(),
                     "Tensor is OutSide the device, can't get the memory.");
        return m_data;
    };

    template <typename T>
    T* ptr() {
        INFER_ASSERT(is_own(),
                     "Tensor is OutSide the device, can't get the memory.");
        return static_cast<T*>(m_data);
    };

    virtual void set_shared_memory(void* data, size_t length = 0);

    //! before use the tensor, should call prepare data, if data is not ready,
    //! it will mmap the data or allocate the data
    virtual TensorState prepare_data();

    //! if the data is mmaped, after use it, should call recall_data
    TensorState recall_data();

    int32_t add_user() {
        m_usr_count++;
        return m_usr_count;
    }

    int32_t get_curr_user_count() { return m_cur_count; };
    int32_t decrease_curr_user_count() {
        if (!m_shared) {
            INFER_ASSERT(m_cur_count > 0, "The user count is less than 0.");
            m_cur_count--;
            if (m_cur_count == 0) {
                recall_data();
            }
        }
        return m_cur_count;
    };

    int32_t resume_user_count() {
        m_cur_count = m_usr_count;
        return m_cur_count;
    }

    bool shared() const { return m_shared; }

    void set_file(std::shared_ptr<InputFile> file, size_t offset) {
        m_state = TensorState::OutSide;
        m_file = file;
        m_file_offset = offset;
    }

private:
    bool m_shared = false;
    int32_t m_usr_count = 0;
    int32_t m_cur_count = 0;

    Device* m_device;
    OpBase* m_owner_op;

    //! backup the data of the tensor
    TensorState m_state;
    //! if m_file is not nullptr, the data is mmaped from the file
    std::shared_ptr<InputFile> m_file;
    size_t m_file_offset = 0;

    uint32_t m_dims = 0;
    size_t m_length = 0;
    DType m_dtype;
    std::vector<size_t> m_shape;
    std::vector<size_t> m_stride;
    void* m_data = nullptr;
    std::string m_name;
};

class WorkSpace {
public:
    void* ptr() { return m_data; };

    template <typename T>
    T* ptr() {
        return static_cast<T*>(m_data);
    };

    size_t length() { return m_length; }

    void set_memory(void* data, size_t length) {
        m_data = data;
        m_length = length;
    }

private:
    void* m_data;
    size_t m_length;
};

}  // namespace inferllm
