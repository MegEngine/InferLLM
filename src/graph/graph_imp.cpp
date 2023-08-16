#include "chatGLM.h"
#include "ggml_llama.h"
#include "llama_like.h"

using namespace inferllm;
std::shared_ptr<Graph> Graph::make_graph(
        UserConfig model_config, Device* device, const std::string& name) {
    if (name == "llama") {
        return std::make_shared<GgmlLlamaGraph>(model_config, device, name);
    } else if (name == "chatglm") {
        return std::make_shared<ChatGLMGraph>(model_config, device, name);
    } else if (name == "chatglm2") {
        return std::make_shared<ChatGLMGraph2>(model_config, device, name);
    } else if (name == "baichuan" || name == "llama2") {
        return std::make_shared<LlamaLikeGraph>(model_config, device, name);
    } else {
        INFER_ASSERT(0, "unsupported model.");
    }
}