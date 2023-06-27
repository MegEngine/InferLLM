#pragma once

#include "llama.h"
#include "chatGLM.h"

namespace inferllm {

class BaiChuanGraph : public Graph {
    using Graph::Graph;

public:
    void set_weights_alias() override;
    void construct_llm() override;
    uint32_t get_nr_ctx() override { return m_param.n_ctx; }
    uint32_t get_nr_vocab() override { return m_param.n_vocab; }
    void load(
            std::shared_ptr<InputFile> fin, LlmParams& param,
            std::shared_ptr<Vocab> vocab) override;

    LlmParams m_param;
};
}  // namespace inferllm