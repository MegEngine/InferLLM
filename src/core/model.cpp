#include "model.h"
#include "model_imp.h"

using namespace inferllm;

Model::Model(const ModelConfig& config, const std::string& model_name) {
    //! TODO: create the model implement by the model name
    m_model_imp = std::make_shared<ModelImp>(config, model_name);
}

void Model::load(const std::string& model_path) {
    m_model_imp->load(model_path);
}

//! allocate memory for the model
void Model::init(uint32_t top_k, float top_p, float temp,
                 float repeat_penalty , int repeat_last_n, int32_t seed) {
    m_model_imp->init(top_k, top_p, temp, repeat_penalty, repeat_last_n, seed);
}

uint32_t Model::get_remain_token() {
    return m_model_imp->get_remain_token();
}

void Model::reset_token() {
    return m_model_imp->reset_token();
}

//! prefill the model with inference with the given promote
void Model::prefill(const std::string& promote) {
    return m_model_imp->prefill(promote);
}

//! decode the answer one by one
std::string Model::decode(const std::string& user_input, int& token) {
    return m_model_imp->decode(user_input, token);
}

//! decode the answer one by one
std::string Model::decode_iter(int& token) {
    return m_model_imp->decode_iter(token);
}
