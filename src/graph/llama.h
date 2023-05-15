#pragma once

#include <unordered_map>
#include "core/kvstorage.h"
#include "core/op.h"
#include "core/tensor.h"
#include "core/graph.h"

namespace inferllm {

enum class LlamaModelType {
    LLAMA_FILE_VERSION_GGML = 0,
    LLAMA_FILE_VERSION_GGMF_V1,  // added version field and scores in vocab
    LLAMA_FILE_VERSION_GGJT_V1
};

class LlamaGraph : public Graph {
    using Graph::Graph;

public:
    void set_weights_alias() override;
    void constuct_llm() override;
    uint32_t get_nr_ctx() override { return m_param.n_ctx; }
    uint32_t get_nr_vocab() override { return m_param.n_vocab; }
    void load(std::shared_ptr<InputFile> fin, LlmParams& param,
              std::shared_ptr<Vocab> vocab) override;

private:
    LlmParams m_param;
};
}  // namespace inferllm