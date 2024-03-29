#include "checker.h"
#include "core/op.h"

namespace inferllm {
namespace test {

//! specializations create Opr
template <>
template <>
void Checker<Embedding>::create_opr(uint32_t embd, uint32_t vocab, DType compt_type) {
    m_naive_values.clear();
    m_device_values.clear();
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

template <>
template <>
void Checker<Elemwise>::create_opr(ElemMode mode, float scale) {
    // if uary
    m_naive_values.clear();
    m_device_values.clear();
    if (mode == ElemMode::Gelu || mode == ElemMode::Silu) {
        auto naive_input = std::make_shared<Tensor>(m_naive_device, "naive_input");
        auto device_input = std::make_shared<Tensor>(m_device, "device_input");

        m_device_opr = std::make_shared<Elemwise>(
                m_device, "device_opr", OpIOs{device_input}, mode, scale);
        m_naive_opr = std::make_shared<Elemwise>(
                m_naive_device, "naive_opr", OpIOs{naive_input}, mode, scale);

        m_naive_values.push_back(naive_input);
        m_device_values.push_back(device_input);
        // else binary
    } else {
        auto naive_input0 = std::make_shared<Tensor>(m_naive_device, "naive_input0");
        auto naive_input1 = std::make_shared<Tensor>(m_naive_device, "naive_input1");
        auto device_input0 = std::make_shared<Tensor>(m_device, "device_input0");
        auto device_input1 = std::make_shared<Tensor>(m_device, "device_input1");

        m_device_opr = std::make_shared<Elemwise>(
                m_device, "device_opr", OpIOs{device_input0, device_input1}, mode,
                scale);
        m_naive_opr = std::make_shared<Elemwise>(
                m_naive_device, "naive_opr", OpIOs{naive_input0, naive_input1}, mode,
                scale);

        m_naive_values.push_back(naive_input0);
        m_naive_values.push_back(naive_input1);
        m_device_values.push_back(device_input0);
        m_device_values.push_back(device_input1);
    }
    m_naive_output = m_naive_opr->outputs()[0];
    m_device_output = m_device_opr->outputs()[0];
}

template <>
template <>
void Checker<LayerNorm>::create_opr(
        int embd, bool mul, bool bias, bool rms, float eps) {
    m_naive_values.clear();
    m_device_values.clear();
    auto naive_input = std::make_shared<Tensor>(m_naive_device, "naive_input");
    auto device_input = std::make_shared<Tensor>(m_device, "device_input");

    m_device_opr = std::make_shared<LayerNorm>(
            m_device, "device_opr", OpIOs{device_input}, embd, mul, bias, rms, eps);
    m_naive_opr = std::make_shared<LayerNorm>(
            m_naive_device, "naive_opr", OpIOs{naive_input}, embd, mul, bias, rms, eps);

    m_naive_values.push_back(naive_input);
    m_device_values.push_back(device_input);

    m_device_weights = m_device_opr->weights();
    m_naive_weights = m_naive_opr->weights();

    m_naive_output = m_naive_opr->outputs()[0];
    m_device_output = m_device_opr->outputs()[0];
}

template <>
template <>
void Checker<MatMul>::create_opr(size_t N, size_t K, bool bias) {
    m_naive_values.clear();
    m_device_values.clear();
    auto naive_input = std::make_shared<Tensor>(m_naive_device, "naive_input");
    auto device_input = std::make_shared<Tensor>(m_device, "device_input");

    m_device_opr = std::make_shared<MatMul>(
            m_device, "device_opr", OpIOs{device_input}, std::vector<size_t>{N, K},
            bias);
    m_naive_opr = std::make_shared<MatMul>(
            m_naive_device, "naive_opr", OpIOs{naive_input}, std::vector<size_t>{N, K},
            bias);

    m_naive_values.push_back(naive_input);
    m_device_values.push_back(device_input);

    m_device_weights = m_device_opr->weights();
    m_naive_weights = m_naive_opr->weights();

    m_naive_output = m_naive_opr->outputs()[0];
    m_device_output = m_device_opr->outputs()[0];
}

#define NO_PARAM_CREATOR(Op)                                                           \
    template <>                                                                        \
    template <>                                                                        \
    void Checker<Op>::create_opr() {                                                   \
        m_naive_values.clear();                                                        \
        m_device_values.clear();                                                       \
        auto naive_input = std::make_shared<Tensor>(m_naive_device, "naive_input");    \
        auto device_input = std::make_shared<Tensor>(m_device, "device_input");        \
        m_device_opr =                                                                 \
                std::make_shared<Op>(m_device, "device_opr", OpIOs{device_input});     \
        m_naive_opr =                                                                  \
                std::make_shared<Op>(m_naive_device, "naive_opr", OpIOs{naive_input}); \
        m_naive_values.push_back(naive_input);                                         \
        m_device_values.push_back(device_input);                                       \
        m_device_weights = m_device_opr->weights();                                    \
        m_naive_weights = m_naive_opr->weights();                                      \
        m_naive_output = m_naive_opr->outputs()[0];                                    \
        m_device_output = m_device_opr->outputs()[0];                                  \
    }

NO_PARAM_CREATOR(SoftMax)
NO_PARAM_CREATOR(DiagMask)

#define AttentionDefine(Op, fuse_weight)                                              \
    template <>                                                                       \
    template <>                                                                       \
    void Checker<Op>::create_opr(                                                     \
            uint32_t embd, uint32_t rot, uint32_t nr_ctx, uint32_t head,              \
            uint32_t layer_id, DType compt_type, RotMode rot_mode) {                  \
        m_naive_values.clear();                                                       \
        m_device_values.clear();                                                      \
        auto naive_input = std::make_shared<Tensor>(m_naive_device, "naive_input");   \
        auto device_input = std::make_shared<Tensor>(m_device, "device_input");       \
        m_device_opr = std::make_shared<Op>(                                          \
                m_device, "device_opr", OpIOs{device_input}, embd, rot, nr_ctx, head, \
                layer_id, compt_type, fuse_weight, false, rot_mode);                  \
        m_naive_opr = std::make_shared<Op>(                                           \
                m_naive_device, "naive_opr", OpIOs{naive_input}, embd, rot, nr_ctx,   \
                head, layer_id, compt_type, fuse_weight, false, rot_mode);            \
        m_naive_values.push_back(naive_input);                                        \
        m_device_values.push_back(device_input);                                      \
        m_device_weights = m_device_opr->weights();                                   \
        m_naive_weights = m_naive_opr->weights();                                     \
        m_naive_output = m_naive_opr->outputs()[0];                                   \
        m_device_output = m_device_opr->outputs()[0];                                 \
    }

AttentionDefine(LlamaAttention, false);
AttentionDefine(GlmAttention, false);
AttentionDefine(Glm2MultiQueryAttention, true);

}  // namespace test
}  // namespace inferllm