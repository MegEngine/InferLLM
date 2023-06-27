#include "checker.h"
#include <math.h>
#include "float.h"
#include "kern/naive/quantize.h"

using namespace inferllm;
namespace {
static int equal_shape(
        const std::vector<size_t>& shape0, const std::vector<size_t>& shape1) {
    if (shape0.size() != shape1.size()) {
        return false;
    }
    for (int32_t i = 0; i < shape0.size(); i++) {
        if (shape0[i] != shape1[i]) {
            return false;
        }
    }
    return true;
}

static std::string shape_to_string(const std::vector<size_t>& shape) {
    std::stringstream ss;
    ss << "shape: dims[";
    for (int32_t i = 0; i < shape.size(); i++) {
        ss << shape[i] << ",";
    }
    ss << "]";
    return ss.str();
}

static inline float diff(float x, float y) {
    return x - y;
}
static inline int diff(int x, int y) {
    return x - y;
}

static inline bool good_float(float val) {
    return std::isfinite(val);
}

static inline bool good_float(int) {
    return true;
}

template <typename ctype>
::testing::AssertionResult assert_tensor_eq_with_dtype(
        const char* expr0, const char* expr1, const Tensor& v0, const Tensor& v1,
        float maxerr, float maxerr_avg, float maxerr_avg_biased) {
    size_t nr_elem = 1;
    for (int i = 0; i < v0.shape().size(); ++i) {
        nr_elem *= v0.shape()[i];
    }
    double error_sum = 0;
    double error_sum_biased = 0;
    const ctype* ptr0 = static_cast<const ctype*>(v0.ptr());
    const ctype* ptr1 = static_cast<const ctype*>(v1.ptr());
    for (size_t i = 0; i < nr_elem; ++i) {
        ctype iv0 = ptr0[i], iv1 = ptr1[i];
        float err = diff(iv0, iv1);
        error_sum += std::abs(err);
        error_sum_biased += err;
        if (!good_float(iv0) || !good_float(iv1) || std::abs(err) > maxerr) {
            return ::testing::AssertionFailure()
                << "Unequal value\n"
                << "Value of: " << expr1 << "\n"
                << "  Actual: " << (iv1 + 0) << "\n"
                << "Expected: " << expr0 << "\n"
                << "Which is: " << (iv0 + 0) << "\n"
                << "At index: " << i << "\n"
                << "tensor v0 shape : " << shape_to_string(v0.shape()) << "\n"
                << "tensor v1 shape : " << shape_to_string(v1.shape()) << "\n";
        }
    }
    float error_avg = error_sum / nr_elem;
    if (error_avg > maxerr_avg) {
        return ::testing::AssertionFailure()
            << "Average error exceeds the upper limit\n"
            << "Value of: " << expr1 << "\n"
            << "Expected: " << expr0 << "\n"
            << "Average error: " << error_avg << "/" << maxerr_avg << "\n"
            << "Num of elements: " << nr_elem;
    }
    float error_avg_biased = error_sum_biased / nr_elem;
    if (std::abs(error_avg_biased) > maxerr_avg_biased) {
        return ::testing::AssertionFailure()
            << "Average biased error exceeds the upper limit\n"
            << "Value of: " << expr1 << "\n"
            << "Expected: " << expr0 << "\n"
            << "Average biased error: " << error_avg_biased << "/" << maxerr_avg_biased
            << "\n"
            << "Num of elements: " << nr_elem;
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult assert_tensor_eq(
        const char* expr0, const char* expr1, const char* /*expr_maxerr*/,
        const char* /*expr_maxerr_avg*/, const char* /*expr_maxerr_avg*/,
        const Tensor& v0, const Tensor& v1, float maxerr, float maxerr_avg,
        float maxerr_avg_biased) {
    if (!equal_shape(v0.shape(), v1.shape())) {
        return ::testing::AssertionFailure()
            << "Shape mismatch\n"
            << "Value of: " << expr1 << "\n"
            << "  Actual: " << shape_to_string(v1.shape()) << "\n"
            << "Expected: " << expr0 << "\n"
            << "Which is: " << shape_to_string(v0.shape()) << "\n";
    }
    if (v0.dtype() != v1.dtype()) {
        return ::testing::AssertionFailure() << "Dtype mismatch \n";
    }
    switch (v0.dtype()) {
#define CASE(enum_, ctype_)                                                   \
    case enum_: {                                                             \
        return assert_tensor_eq_with_dtype<ctype_>(                           \
                expr0, expr1, v0, v1, maxerr, maxerr_avg, maxerr_avg_biased); \
    }
        CASE(DType::Float32, float)
        CASE(DType::Int32, int)
        CASE(DType::Int8, int8_t)
        default:
            printf("unsupport dtype in check tensor equal.");
            abort();
#undef CASE
    }
}

}  // namespace

namespace inferllm {
namespace test {

class RandomState {
public:
    static std::mt19937& generator() { return instance()->m_generator; }

    static void reset() { instance()->m_generator.seed(42); }

private:
    RandomState() : m_generator(42) {}
    std::mt19937 m_generator;
    static RandomState* instance() { return &m_instance; }
    static RandomState m_instance;
};

RandomState RandomState::m_instance;

void check_tensor(
        const Tensor& computed, const Tensor& expected, float epsilon,
        float max_avg_error, float max_avg_biased_error) {
    if (expected.shape().size() == 0)
        return;
    ASSERT_TENSOR_EQ_EPS_AVG(
            computed, expected, epsilon, max_avg_error, max_avg_biased_error);
}

void IIDRNG::gen(Tensor& tensor) {
    if (tensor.dtype() == DType::Float32 && has_fast_float32()) {
        fill_fast_float32(tensor.ptr<float>(), tensor.length());
        return;
    } else if (tensor.dtype() == DType::Int4) {
        std::vector<float> tmp(tensor.length());
        fill_fast_float32(tmp.data(), tensor.length());
        naive::quantize_row_q4_0_reference(
                tmp.data(), (BlockQ40*)tensor.ptr(), tensor.length());
        return;
    } else if (tensor.dtype() == DType::Int32) {
        for(size_t i = 0; i < tensor.length(); ++i) {
            tensor.ptr<int32_t>()[i] = gen_single_val();
        }
        return;
    } else {
        INFER_ASSERT(0, "Not implemented.");
    }
}

bool IIDRNG::has_fast_float32() {
    return false;
}

void IIDRNG::fill_fast_float32(float*, size_t) {
    INFER_ASSERT(0, "Not implemented.");
}

float NormalRNG::gen_single_val() {
    auto&& gen = RandomState::generator();
    return m_dist(gen);
}

bool NormalRNG::has_fast_float32() {
    return true;
}

void NormalRNG::fill_fast_float32(float* dest, size_t size) {
    auto gen = RandomState::generator();
    for (size_t i = 0; i < size; ++i) {
        dest[i] = m_dist(gen);
    }
}

uint32_t UIntRNG::gen_single_val() {
    auto&& gen = RandomState::generator();
    return m_dist(gen);
}

//! specializations create Opr
template <>
template <>
void Checker<Embedding>::create_opr(uint32_t embd, uint32_t vocab, DType compt_type) {
    auto naive_input = std::make_shared<Tensor>(m_naive_device, "naive_input");
    auto device_input = std::make_shared<Tensor>(m_device, "device_input");
    m_device_opr = std::make_shared<Embedding>(
            OpIOs{device_input}, embd, vocab, compt_type, m_device, "device_opr");
    m_naive_opr = std::make_shared<Embedding>(
            OpIOs{naive_input}, embd, vocab, compt_type, m_naive_device, "naive_opr");
    m_naive_values.push_back(naive_input);
    m_naive_weights.push_back(m_naive_opr->weights()[0]);

    m_device_values.push_back(device_input);
    m_device_weights.push_back(m_device_opr->weights()[0]);

    m_naive_output = m_naive_opr->outputs()[0];
    m_device_output = m_device_opr->outputs()[0];
}

}  // namespace test
}  // namespace inferllm