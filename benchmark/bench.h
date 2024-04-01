#ifndef BENCHMARK_BENCH_H_
#define BENCHMARK_BENCH_H_

#include <sys/time.h>

#include <vector>

#include "core/op.h"
#include "core/tensor.h"
#include "utils.h"

#pragma once
#include <memory>
#include <regex>
#include <unordered_map>

#include "file.h"

namespace llm_learning {
namespace benchmark {
using TensorShape = std::vector<size_t>;

class BenchmarkHelper {
 public:
  using TensorValueArray = std::vector<std::shared_ptr<Tensor>>;
  using TensorShapeArray = std::vector<std::vector<size_t>>;

  Device* device() const { return m_device; }

  BenchmarkHelper() {}

 protected:
  Device* m_device;
  std::unique_ptr<RNG> m_default_rng;
  std::unordered_map<size_t, RNG*> m_rng;
  std::unordered_map<size_t, RNG*> m_weight_rng;
  std::unordered_map<size_t, DType> m_dtype;
  std::unordered_map<size_t, DType> m_weight_dtype;
  float_t m_epsilon = 1e-3;
  int32_t m_past = 0;

  BenchmarkHelper(Device* device) {
    m_device = device;
    m_default_rng = std::unique_ptr<RNG>(new NormalRNG());
  }

  ~BenchmarkHelper() noexcept = default;
};

template <typename Opr>
class Benchmark : public BenchmarkHelper {
 public:
  Benchmark(Device* device) : BenchmarkHelper(device) {}

  void exec(const TensorShapeArray& shapes);

  Benchmark& set_dtype(size_t idx, DType dtype) {
    m_dtype[idx] = dtype;
    return *this;
  }
  Benchmark& set_weight_dtype(size_t idx, DType dtype) {
    m_weight_dtype[idx] = dtype;
    return *this;
  }
  Benchmark& set_rng(size_t idx, RNG* rng) {
    m_rng[idx] = rng;
    return *this;
  }
  Benchmark& set_weight_rng(size_t idx, RNG* rng) {
    m_weight_rng[idx] = rng;
    return *this;
  }
  //! max error of a single element
  Benchmark& set_epsilon(float epsilon) {
    m_epsilon = epsilon;
    return *this;
  }

  Benchmark& set_past(int32_t past) {
    m_past = past;
    return *this;
  }

  Benchmark& weight_packed(bool packed) {
    m_weight_packed = packed;
    return *this;
  }

  template <typename... Args>
  void create_opr(Args... args);

 private:
  std::shared_ptr<Opr> m_opr;
  TensorValueArray m_values;
  TensorValueArray m_weights;

  std::shared_ptr<Tensor> m_output;
  bool m_weight_packed = false;
};

template <typename Opr>
void Benchmark<Opr>::exec(const TensorShapeArray& shapes) {
  //! set tensor shape
  INFER_ASSERT(shapes.size() == m_values.size(),
               "The number of input shapes is not equal to the number of inputs.");

  auto init_input = [this, &shapes](BenchmarkHelper::TensorValueArray& inputs, int& index) {
    for (auto input : inputs) {
      input->set_shape(shapes[index]);
      if (m_dtype.find(index) != m_dtype.end()) {
        input->set_dtype(m_dtype[index]);
      } else {
        input->set_dtype(DType::Float32);
      }
      input->prepare_data();
      index++;
    }
  };
  int index = 0;
  init_input(m_values, index);

  auto init_weight = [this, &shapes](BenchmarkHelper::TensorValueArray& weights, int& index) {
    for (auto weight : weights) {
      if (m_weight_dtype.find(index) != m_weight_dtype.end()) {
        weight->set_dtype(m_weight_dtype[index]);
      } else {
        weight->set_dtype(DType::Float32);
      }
      weight->prepare_data();
      index++;
    }
  };
  index = 0;
  init_weight(m_weights, index);

  index = 0;
  for (auto input : m_values) {
    if (m_rng.find(index) != m_rng.end()) {
      m_rng[index]->gen(*input);
    } else {
      m_default_rng->gen(*input);
    }
    index++;
  }
  
  index = 0;
  for (auto weight : m_weights) {
    if (m_weight_rng.find(index) != m_weight_rng.end()) {
      m_weight_rng[index]->gen(*weight);
    } else {
      m_default_rng->gen(*weight);
    }
    index++;
  }

  if (m_weight_packed) {
    for (auto weight : m_weights) {
      weight->preprocess_data();
    }
  }
  //! allocate output memory and workspace
  m_output->prepare_data();

  auto device_size = m_opr->get_workspace_in_byte();
  auto device_ptr = m_device->allocate(device_size);
  auto device_workspace = std::make_shared<WorkSpace>();
  device_workspace->set_memory(device_ptr, device_size);
  //! exec opr
  m_opr->execute(device_workspace.get(), m_past);

  m_device->free_device(device_ptr);

  for (auto input : m_values) {
    input->recall_data();
  }
  for (auto weight : m_weights) {
    weight->recall_data();
  }
  m_output->recall_data();
}

}  // namespace benchmark
}  // namespace llm_learning

#endif