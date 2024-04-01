#include "bench.h"
#include "core/op.h"

namespace llm_learning {
namespace benchmark {

//! specializations create Opr
template <>
template <>
void Benchmark<Embedding>::create_opr(uint32_t embd, uint32_t vocab, DType compt_type) {
  m_values.clear();
  m_weights.clear();
  auto input = std::make_shared<Tensor>(m_device, "Embedding_input");
  m_opr =
      std::make_shared<Embedding>(OpIOs{input}, embd, vocab, compt_type, m_device, "Embedding_opr");
  m_values.push_back(input);
  m_weights.push_back(m_opr->weights()[0]);

  m_output = m_opr->outputs()[0];
}

//! specializations create Opr
template <>
template <>
void Benchmark<SoftMax>::create_opr() {
  m_values.clear();
  m_weights.clear();
  auto input = std::make_shared<Tensor>(m_device, "SoftMax_input");
  m_opr = std::make_shared<SoftMax>(m_device, "SoftMax_opr", OpIOs{input});
  m_values.push_back(input);

  m_output = m_opr->outputs()[0];
}

}  // namespace benchmark
}  // namespace llm_learning