#pragma once

#include "core/graph.h"

namespace inferllm {

class LlamaLikeGraph : public Graph {
    using Graph::Graph;

public:
    void set_weights_alias() override;
    void construct_llm() override;
    void load_param(
            std::shared_ptr<InputFile> fin, LlmParams& param,
            std::shared_ptr<Vocab> vocab) override;

};
}  // namespace inferllm