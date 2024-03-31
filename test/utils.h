#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include <gtest/gtest.h>

#include "core/op.h"
#include "model.h"

#pragma once
#include <memory>
#include <regex>
#include <unordered_map>

#include "file.h"

namespace llm_learning {
namespace test {

#include <random>
#include <set>

#if __cplusplus >= 201703L
#define COMPAT_RANDOM(begin, end)          \
  {                                        \
    std::default_random_engine rng_engine; \
    std::shuffle(begin, end, rng_engine);  \
  }
#else
#define COMPAT_RANDOM(begin, end) std::random_shuffle(begin, end);
#endif

#define ASSERT_TENSOR_EQ_EPS_AVG(v0, v1, maxerr, maxerr_avg, maxerr_avg_biased) \
  ASSERT_PRED_FORMAT5(assert_tensor_eq, v0, v1, maxerr, maxerr_avg, maxerr_avg_biased)

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
  NormalRNG(float mean = 0.0f, float stddev = 1.0f) : m_dist(mean, stddev) {}

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

}  // namespace test
}  // namespace llm_learning

#endif