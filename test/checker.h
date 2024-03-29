#pragma once

#include "model.h"
#include "core/op.h"
#include <gtest/gtest.h>

#pragma once
#include <memory>
#include <regex>
#include <unordered_map>
#include "file.h"

namespace inferllm {
namespace test {

using TensorShape = std::vector<size_t>;

#include <random>
#include <set>

#if __cplusplus >= 201703L
#define COMPAT_RANDOM(begin, end)              \
    {                                          \
        std::default_random_engine rng_engine; \
        std::shuffle(begin, end, rng_engine);  \
    }
#else
#define COMPAT_RANDOM(begin, end) std::random_shuffle(begin, end);
#endif

class RNG {
protected:
    class RNGxorshf;

public:
    virtual void gen(Tensor& tensor) = 0;
    virtual ~RNG() = default;
};

class IIDRNG : public RNG {
public:
    void gen(Tensor& tensor) override;
    virtual float gen_single_val() = 0;
    virtual bool output_is_float() { return true; }

protected:
    virtual bool has_fast_float32();
    virtual void fill_fast_float32(float* dest, size_t size);
};

class NormalRNG final : public IIDRNG {
public:
    NormalRNG(float mean = 0.0f, float stddev = 1.0f)
            : m_dist(mean, stddev) {}

    void fill_fast_float32(float* dest, size_t size) override;

protected:
    float gen_single_val() override;

private:
    std::normal_distribution<float> m_dist;
    bool has_fast_float32() override;
};

class UIntRNG final : public RNG {
public:
    UIntRNG(int min, int max) {
        m_min = min;
        m_max = max;
        m_dist = std::uniform_int_distribution<uint32_t>(min, max);
    }

    void gen(Tensor& tensor) override {
        auto* ptr = tensor.ptr<int>();
        for (size_t i = 0; i < tensor.length(); i++) {
            ptr[i] = gen_single_val();
        }
    }

    uint32_t gen_single_val();

private:
    uint32_t m_min;
    uint32_t m_max;
    std::uniform_int_distribution<uint32_t> m_dist;
};

class CheckerHelper {
public:
    using TensorValueArray = std::vector<std::shared_ptr<Tensor>>;
    using TensorShapeArray = std::vector<std::vector<size_t>>;

    Device* device() const { return m_naive_device; }

    CheckerHelper() {}

protected:
    Device* m_naive_device;
    Device* m_device;
    std::unique_ptr<RNG> m_default_rng;
    std::unordered_map<size_t, RNG*> m_rng;
    std::unordered_map<size_t, RNG*> m_weight_rng;
    std::unordered_map<size_t, DType> m_dtype;
    std::unordered_map<size_t, DType> m_weight_dtype;
    float_t m_epsilon = 1e-3;
    int32_t m_past = 0;

    CheckerHelper(Device* device, Device* naive) {
        m_device = device;
        m_naive_device = naive;
        m_default_rng = std::unique_ptr<RNG>(new NormalRNG());
    }

    ~CheckerHelper() noexcept = default;
};

void check_tensor(
        const Tensor& computed, const Tensor& expected, float epsilon = 1e-3,
        float max_avg_error = 1e-1, float max_avg_biased_error = 1e-1);

#define ASSERT_TENSOR_EQ_EPS_AVG(v0, v1, maxerr, maxerr_avg, maxerr_avg_biased) \
    ASSERT_PRED_FORMAT5(assert_tensor_eq, v0, v1, maxerr, maxerr_avg, maxerr_avg_biased)

template <typename Opr>
class Checker : public CheckerHelper {
public:
    Checker(Device* device, Device* naive) : CheckerHelper(device, naive) {}

    float exec(const TensorShapeArray& shapes);

    Checker& set_dtype(size_t idx, DType dtype) {
        m_dtype[idx] = dtype;
        return *this;
    }
    Checker& set_weight_dtype(size_t idx, DType dtype) {
        m_weight_dtype[idx] = dtype;
        return *this;
    }
    Checker& set_rng(size_t idx, RNG* rng) {
        m_rng[idx] = rng;
        return *this;
    }
    Checker& set_weight_rng(size_t idx, RNG* rng) {
        m_weight_rng[idx] = rng;
        return *this;
    }
    //! max error of a single element
    Checker& set_epsilon(float epsilon) {
        m_epsilon = epsilon;
        return *this;
    }

    Checker& set_past(int32_t past) {
        m_past = past;
        return *this;
    }

    Checker& weight_packed(bool packed) {
        m_weight_packed = packed;
        return *this;
    }

    Checker& profile(size_t times){
        m_profile = true;
        m_profile_times = times;
        return *this;
    }

    template <typename... Args>
    void create_opr(Args... args);

private:
    std::shared_ptr<Opr> m_naive_opr;
    std::shared_ptr<Opr> m_device_opr;
    TensorValueArray m_naive_values;
    TensorValueArray m_device_values;

    TensorValueArray m_naive_weights;
    TensorValueArray m_device_weights;

    std::shared_ptr<Tensor> m_naive_output;
    std::shared_ptr<Tensor> m_device_output;
    std::shared_ptr<Tensor> m_deivce_output_CPU;

    bool m_weight_packed = false;
    bool m_profile = false;
    bool m_profile_times = 1;
    double m_timeuse = 0.0;
};

template <typename Opr>
float Checker<Opr>::exec(const TensorShapeArray& shapes) {
    //! set tensor shape
    INFER_ASSERT(
            shapes.size() == m_naive_values.size(),
            "The number of input shapes is not equal to the number of inputs.");
    INFER_ASSERT(
            shapes.size() == m_device_values.size(),
            "The number of input shapes is not equal to the number of inputs.");
    int index = 0;
    for (auto input : m_naive_values) {
        input->set_shape(shapes[index]);
        if (m_dtype.find(index) != m_dtype.end()) {
            input->set_dtype(m_dtype[index]);
        } else {
            input->set_dtype(DType::Float32);
        }
        input->prepare_data();
        index++;
    }
    index = 0;
    for (auto input : m_device_values) {
        input->set_shape(shapes[index]);
        if (m_dtype.find(index) != m_dtype.end()) {
            input->set_dtype(m_dtype[index]);
        } else {
            input->set_dtype(DType::Float32);
        }
        input->prepare_data();
        index++;
    }
    index = 0;
    for (auto weight : m_naive_weights) {
        if (m_weight_dtype.find(index) != m_weight_dtype.end()) {
            weight->set_dtype(m_weight_dtype[index]);
        } else {
            weight->set_dtype(DType::Float32);
        }
        weight->prepare_data();
        index++;
    }
    index = 0;
    for (auto weight : m_device_weights) {
        if (m_weight_dtype.find(index) != m_weight_dtype.end()) {
            weight->set_dtype(m_weight_dtype[index]);
        } else {
            weight->set_dtype(DType::Float32);
        }
        weight->prepare_data();
        index++;
    }
    //! generate random value
    //! copy naive value to device
    auto host2device = [this](std::shared_ptr<Tensor> dest,
                              std::shared_ptr<Tensor> src) {
        m_device->host2device_copy(dest->ptr(), src->ptr(), src->length_in_byte());
    };
    index = 0;
    for (auto input : m_naive_values) {
        if (m_rng.find(index) != m_rng.end()) {
            m_rng[index]->gen(*input);
        } else {
            m_default_rng->gen(*input);
        }
        host2device(m_device_values[index], input);
        index++;
    }
    index = 0;
    for (auto weight : m_naive_weights) {
        if (m_weight_rng.find(index) != m_weight_rng.end()) {
            m_weight_rng[index]->gen(*weight);
        } else {
            m_default_rng->gen(*weight);
        }
        host2device(m_device_weights[index], weight);
        index++;
    }

    if (m_weight_packed) {
        for (auto weight : m_naive_weights) {
            weight->preprocess_data();
        }
        for (auto weight : m_device_weights) {
            weight->preprocess_data();
        }
    }
    //! deduce output shape
    m_deivce_output_CPU = std::make_shared<Tensor>(m_naive_device, "deivce_output_CPU");
    m_naive_opr->deduce_output_shape();
    m_device_opr->deduce_output_shape();
    m_deivce_output_CPU->set_shape(m_naive_output->shape());
    m_deivce_output_CPU->set_dtype(m_naive_output->dtype());
    //! allocate output memory and workspace
    m_naive_output->prepare_data();
    m_device_output->prepare_data();
    m_deivce_output_CPU->prepare_data();

    auto naive_size = m_naive_opr->get_workspace_in_byte();
    auto naive_ptr = m_naive_device->allocate(naive_size);
    auto naive_workspace = std::make_shared<WorkSpace>();
    naive_workspace->set_memory(naive_ptr, naive_size);

    auto device_size = m_device_opr->get_workspace_in_byte();
    auto device_ptr = m_device->allocate(device_size);
    auto device_workspace = std::make_shared<WorkSpace>();
    device_workspace->set_memory(device_ptr, device_size);
    //! exec opr
    m_naive_opr->execute(naive_workspace.get(), m_past);
    m_device_opr->execute(device_workspace.get(), m_past);

    if (m_profile) {
        // warmup
        for (size_t i = 0; i < 5; i++) {
            m_device_opr->execute(naive_workspace.get(), m_past);
        }
        struct timeval start, end;
        gettimeofday(&start, NULL);
        for (size_t i = 0; i < m_profile_times; i++) {
            m_device_opr->execute(device_workspace.get(), m_past);
        }
        gettimeofday(&end, NULL);
        m_timeuse =
                (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    }
    //! copy device value to host
    m_device->device2host_copy(
            m_deivce_output_CPU->ptr(), m_device_output->ptr(),
            m_device_output->length_in_byte());
    //! check result
    check_tensor(*m_deivce_output_CPU, *m_naive_output, m_epsilon);
    m_naive_device->free_device(naive_ptr);
    m_device->free_device(device_ptr);

    for (auto input : m_naive_values) {
        input->recall_data();
    }
    for (auto input : m_device_values) {
        input->recall_data();
    }
    for (auto weight : m_naive_weights) {
        weight->recall_data();
    }
    for (auto weight : m_device_weights) {
        weight->recall_data();
    }
    m_naive_output->recall_data();
    m_device_output->recall_data();
    m_deivce_output_CPU->recall_data();
    return m_timeuse;
}

}  // namespace test
}  // namespace inferllm