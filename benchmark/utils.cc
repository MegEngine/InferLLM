
#include "utils.h"
namespace llm_learning {
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

void IIDRNG::gen(Tensor& tensor) {
  if (tensor.dtype() == DType::Float32 && has_fast_float32()) {
    fill_fast_float32(tensor.ptr<float>(), tensor.length());
    return;
  } else if (tensor.dtype() == DType::Int4) {
    std::vector<float> tmp(tensor.length());
    fill_fast_float32(tmp.data(), tensor.length());
    naive::quantize_row_q4_0_reference(tmp.data(), (BlockQ40*)tensor.ptr(), tensor.length());
    return;
  } else if (tensor.dtype() == DType::Int32) {
    for (size_t i = 0; i < tensor.length(); ++i) {
      tensor.ptr<int32_t>()[i] = gen_single_val();
    }
    return;
  } else {
    INFER_ASSERT(0, "Not implemented.");
  }
}

bool IIDRNG::has_fast_float32() { return false; }

void IIDRNG::fill_fast_float32(float*, size_t) { INFER_ASSERT(0, "Not implemented."); }

float NormalRNG::gen_single_val() {
  auto&& gen = RandomState::generator();
  return m_dist(gen);
}

bool NormalRNG::has_fast_float32() { return true; }

void NormalRNG::fill_fast_float32(float* dest, size_t size) {
  auto gen = RandomState::generator();
  for (size_t i = 0; i < size; ++i) {
    dest[i] = m_dist(gen);
  }
}

}  // namespace test
}  // namespace llm_learning