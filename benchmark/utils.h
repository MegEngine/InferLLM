#ifndef BENCHMARK_UTILS_H_
#define BENCHMARK_UTILS_H_

#include "core/op.h"
#include "model.h"

#pragma once
#include <memory>
#include <regex>
#include <unordered_map>

#include "file.h"

namespace llm_learning {
namespace benchmark {

#include <random>

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

}  // namespace benchmark
}  // namespace llm_learning

#endif