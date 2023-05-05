#pragma once

#include <memory>
#include <string>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

namespace inferllm {

struct ModelConfig {
    //! dtype include 'float32','float16','int8','int4'
    std::string compt_type;
    uint32_t nr_thread;
    uint32_t nr_ctx;
    bool enable_mmap;
};

class ModelImp;

class API Model {
public:
    //! create a model by the model_name, the model_name must be registered
    //! internal before load it
    Model(const ModelConfig& config, const std::string& model_name);

    //! load the model from model_path
    void load(const std::string& model_path);

    //! allocate memory for the model or init its param
    void init(uint32_t top_k, float top_p, float temp, float repeat_penalty,
              int repeat_last_n, int32_t seed);

    //! get the remain token number
    uint32_t get_remain_token();

    //! reset the token
    void reset_token();

    //! prefill the model with inference with the given promote
    void prefill(const std::string& promote);

    //! decode the answer one by one
    std::string decode(const std::string& user_input, int& token);

    //! decode the answer one by one
    std::string decode_iter(int& token);

private:
    std::shared_ptr<ModelImp> m_model_imp;
};

}  // namespace inferllm
