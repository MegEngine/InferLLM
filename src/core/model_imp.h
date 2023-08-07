#pragma once

#include <list>
#include <map>
#include <memory>
#include <string>

#include "device.h"
#include "graph.h"
#include "kern/kernel_define.h"
#include "model.h"

namespace inferllm {

namespace {

DType dtype_from_str(const std::string& str) {
    if (str == "float32" || str == "fp23") {
        return DType::Float32;
    } else if (str == "float16" || str == "fp16") {
        return DType::Float16;
    } else if (str == "int8") {
        return DType::Int8;
    } else if (str == "uint8") {
        return DType::Uint8;
    } else if (str == "int4") {
        return DType::Int4;
    } else if (str == "uint4") {
        return DType::Uint4;
    } else {
        INFER_ASSERT(0, "Unsupported dytpe.");
    }
}
}  // namespace

//! the implement of model
class ModelImp {
public:
    ModelImp(const ModelConfig& config, const std::string& name)
            : m_name(name), m_config(config) {
        uint32_t nr_thread = config.nr_thread;
        std::string device_type = config.device_type;
        if (device_type == "CPU" || device_type == "cpu") {
#if INFER_X86
            m_device = make_unique<CPUDevice>(KernelType::X86, nr_thread);
#elif INFER_ARM
            m_device = make_unique<CPUDevice>(KernelType::Arm, nr_thread);
#else
            m_device = make_unique<CPUDevice>(KernelType::Naive, nr_thread);
#endif
        } else if (
                device_type == "GPU" || device_type == "CUDA" || device_type == "gpu") {
            // if compile with GPU, use GPU, else use CPUDevice
#if ENABLE_GPU
            m_device = make_unique<GPUDevice>(0);
#else
            INFER_ASSERT(0, "GPU is disabled when build, please build with GPU.");
#endif
        }

        UserConfig user_config;
        user_config.compt_type = dtype_from_str(config.compt_type);
        m_graph = Graph::make_graph(user_config, m_device.get(), name);
        m_past = 0;
    }
    //! load the model from model_path
    void load(const std::string& model_path);

    //! allocate memory for the model or init its param
    void init(
            uint32_t top_k, float top_p, float temp, float repeat_penalty,
            int repeat_last_n, int32_t seed, int32_t end_token) {
        m_top_k = top_k;
        m_top_p = top_p;
        m_temp = temp;
        m_repeat_penalty = repeat_penalty;
        m_repeat_last_n = repeat_last_n;
        m_end_token = end_token;
        for (uint32_t i = 0; i < m_repeat_last_n; i++) {
            m_last_queue.push_back(0);
        }
        m_rng = std::mt19937(seed);
    }

    void active() {
			m_device->active();
		}

    void deactive() {
			m_device->deactive();
		}

    //! prefill the model with inference with the given promote
    void prefill(const std::string& promote);

    //! decode the user input sentence
    std::string decode(const std::string& user_input, int& token);

    std::string decode_iter(int& token);

    uint32_t get_remain_token() { return m_graph->get_nr_ctx() - m_past; }

    void reset_token() {
        m_past = 0;
        m_graph->reset_ctx();
    }

    int32_t sample_and_update();

    std::string decode_summary() const;

private:
    std::vector<Vocab::Id> tokenize(const std::string& text, bool bos);

    uint32_t m_past = 0;

    uint32_t m_top_k;
    float m_top_p;
    float m_temp;
    float m_repeat_penalty;
    uint32_t m_repeat_last_n;
    int32_t m_end_token;

    int32_t m_pre_token;

    std::string m_name;
    LlmParams m_param;
    ModelConfig m_config;
    //! TODO: support run model on multi device later.
    std::unique_ptr<Device> m_device;
    std::shared_ptr<Graph> m_graph;
    std::shared_ptr<Vocab> m_vocab;
    std::list<int32_t> m_last_queue;
    std::vector<float> m_logist;

    std::mt19937 m_rng;
    Timer m_timer;
    double m_time_cost = 0;
};

}  // namespace inferllm
