#include "check.h"
#include "core/op.h"

namespace llm_learning {
namespace test {

//! specializations create Opr
template <>
template <>
void Checker<Embedding>::create_opr(uint32_t embd, uint32_t vocab, DType compt_type) {
  m_naive_values.clear();
  m_device_values.clear();
  auto naive_input = std::make_shared<Tensor>(m_naive_device, "naive_input");
  auto device_input = std::make_shared<Tensor>(m_device, "device_input");
  m_device_opr = std::make_shared<Embedding>(OpIOs{device_input}, embd, vocab, compt_type, m_device,
                                             "device_opr");
  m_naive_opr = std::make_shared<Embedding>(OpIOs{naive_input}, embd, vocab, compt_type,
                                            m_naive_device, "naive_opr");
  m_naive_values.push_back(naive_input);
  m_naive_weights.push_back(m_naive_opr->weights()[0]);

  m_device_values.push_back(device_input);
  m_device_weights.push_back(m_device_opr->weights()[0]);

  m_naive_output = m_naive_opr->outputs()[0];
  m_device_output = m_device_opr->outputs()[0];
}

}  // namespace test
}  // namespace llm_learning