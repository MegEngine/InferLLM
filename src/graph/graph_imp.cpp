#include "graph_imp.h"

using namespace inferllm;
std::shared_ptr<Graph> Graph::make_graph(UserConfig model_config,
                                         Device* device,
                                         const std::string& name) {
    if (name == "llama") {
        return std::make_shared<LlamaGraph>(model_config, device, name);
    } else {
        INFER_ASSERT(0, "unsupported model.");
    }
}