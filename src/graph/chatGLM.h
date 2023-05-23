#pragma once

#include <unordered_map>
#include "core/kvstorage.h"
#include "core/op.h"
#include "core/tensor.h"
#include "core/graph.h"

namespace inferllm {
namespace chatglm {

struct Header {
    int param_offset;
    int param_length;
    int vocab_offset;
    int vocab_length;
    int tensor_offset;
};

struct Param {
    int hidden_size;
    int n_heads;
    int n_layers;
    int embd_dim;
    int fc_hidden;
    int vacab_size;
};

}  // namespace chatglm

class ChatGLMGraph : public Graph {
    using Graph::Graph;

public:
    void set_weights_alias() override;
    void constuct_llm() override;
    uint32_t get_nr_ctx() override { return m_param.n_ctx; }
    uint32_t get_nr_vocab() override { return m_param.n_vocab; }
    void load(std::shared_ptr<InputFile> fin, LlmParams& param,
              std::shared_ptr<Vocab> vocab) override;
    void post_tokenize(std::vector<Vocab::Id>& input) override;

private:
    LlmParams m_param;
};
}  // namespace inferllm