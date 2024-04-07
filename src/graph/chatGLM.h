#pragma once

#include <unordered_map>
#include "core/graph.h"
#include "core/kvstorage.h"
#include "core/op.h"
#include "core/tensor.h"

namespace inferllm {
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
    int multi_query = 0;
    int multi_query_group_num = 1;
};

class ChatGLMGraph : public Graph {
    using Graph::Graph;

public:
    void set_weights_alias() override;
    void construct_llm() override;
    void load_param(
            std::shared_ptr<InputFile> fin, LlmParams& param,
            std::shared_ptr<Vocab> vocab) override;
    void post_tokenize(std::vector<Vocab::Id>& input) override;
};

class ChatGLMGraph2 : public Graph {
    using Graph::Graph;

public:
    void set_weights_alias() override;
    void construct_llm() override;
    void load_param(
            std::shared_ptr<InputFile> fin, LlmParams& param,
            std::shared_ptr<Vocab> vocab) override;

    void post_tokenize(std::vector<Vocab::Id>& input) override;
};

class ChatGLMGraph3 : public Graph {
    using Graph::Graph;

public:
    void set_weights_alias() override;
    void construct_llm() override;
    void load_param(
            std::shared_ptr<InputFile> fin, LlmParams& param,
            std::shared_ptr<Vocab> vocab) override;

    void post_tokenize(std::vector<Vocab::Id>& input) override;
};
}  // namespace inferllm