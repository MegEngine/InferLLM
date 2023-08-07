
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <vector>

#include "Tracy.hpp"
#include "graph.h"

using namespace inferllm;

void OprModuleBase::deduce_output_shape() {
    for (auto opr : m_oprs) {
        opr->deduce_output_shape();
    }
}

void OprModuleBase::execute(WorkSpace* workspace, uint32_t nr_past, bool) {
    ZoneScopedNS("opr", 4);
    for (auto opr : m_oprs) {
        auto sname = opr->name();
        ZoneText(sname.data(), sname.size());
        opr->pre_execute();
#ifdef INFER_PROFILE
        struct timeval start, end;
        gettimeofday(&start, NULL);
#endif
        opr->execute(workspace, nr_past);

#ifdef INFER_PROFILE
        gettimeofday(&end, NULL);
        long seconds = end.tv_sec - start.tv_sec;
        float micros = (seconds * 1000) + (float)(end.tv_usec - start.tv_usec) / 1000;
        printf("Op %s spent time %f ms\n", opr->name().c_str(), micros);
#endif
        opr->end_execute();
    }
}

size_t OprModuleBase::get_workspace_in_byte() {
    size_t max_workspace = 0;
    for (auto opr : m_oprs) {
        size_t workspace = opr->get_workspace_in_byte();
        max_workspace = max_workspace < workspace ? workspace : max_workspace;
    }
    return max_workspace;
}

OprModuleBase::OprModuleBase(
        std::shared_ptr<Tensor> input, Device* device, const std::string& name)
        : m_name(name), m_device(device) {
    m_inputs.push_back(input);
}

OprModuleBase::OprModuleBase(
        std::vector<std::shared_ptr<Tensor>> inputs, Device* device,
        const std::string& name)
        : m_name(name), m_device(device), m_inputs(inputs) {}

LlamaFFNModule::LlamaFFNModule(
        Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t mult,
        UserConfig model_config, Device* device, const std::string& name)
        : OprModuleBase(input, device, name), m_embd(embd), m_graph(graph) {
    size_t nff = ((2 * (4 * embd) / 3 + mult - 1) / mult) * mult;
    //! matmul0
    auto matmul_out0 = add_opr<MatMul>(
            device, name + ".ffn.w3", OpIOs{input}, std::vector<size_t>{nff, embd})[0];
    //! matmul1
    auto matmul_out1 = add_opr<MatMul>(
            device, name + ".ffn.w1", OpIOs{input}, std::vector<size_t>{nff, embd})[0];
    //! silu activation
    auto silu_out = add_opr<Elemwise>(
            device, name + ".silu", OpIOs{matmul_out1}, ElemMode::Silu)[0];
    //! elemwise mul
    auto mul_out = add_opr<Elemwise>(
            device, name + ".elemwise", OpIOs{silu_out, matmul_out0}, ElemMode::Mul)[0];
    //! matmul2
    auto matmul_out2 = add_opr<MatMul>(
            device, name + ".ffn.w2", OpIOs{mul_out},
            std::vector<size_t>{embd, nff})[0];
    set_output(matmul_out2);
}

GlmFFNModule::GlmFFNModule(
        Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t mult,
        UserConfig model_config, Device* device, const std::string& name)
        : OprModuleBase(input, device, name), m_embd(embd), m_graph(graph) {
    //! matmul0
    auto matmul_out1 = add_opr<MatMul>(
            device, name + ".ffn.matmul1", OpIOs{input},
            std::vector<size_t>{mult, embd}, true)[0];
    //! gelu activation
    auto gelu_out = add_opr<Elemwise>(
            device, name + ".gelu", OpIOs{matmul_out1}, ElemMode::Gelu)[0];
    //! matmul2
    auto matmul_out2 = add_opr<MatMul>(
            device, name + ".ffn.matmul2", OpIOs{gelu_out},
            std::vector<size_t>{embd, mult}, true)[0];
    set_output(matmul_out2);
}

Glm2FFNModule::Glm2FFNModule(
        Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t mult,
        UserConfig model_config, Device* device, const std::string& name)
        : OprModuleBase(input, device, name), m_embd(embd), m_graph(graph) {
    //! matmul0
    auto matmul_out1 = add_opr<MatMul>(
            device, name + ".ffn.matmul1", OpIOs{input},
            std::vector<size_t>{mult * 2, embd}, false)[0];
    //! gelu activation
    auto gelu_out = add_opr<SpliteHalfActiveMul>(
            device, name + ".silu", OpIOs{matmul_out1}, ElemMode::Silu)[0];
    //! matmul2
    auto matmul_out2 = add_opr<MatMul>(
            device, name + ".ffn.matmul2", OpIOs{gelu_out},
            std::vector<size_t>{embd, mult}, false)[0];
    set_output(matmul_out2);
}

HeadModule::HeadModule(
        Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t vocab,
        UserConfig model_config, Device* device, const std::string& name, bool bias,
        float eps)
        : OprModuleBase(input, device, name), m_embd(embd), m_graph(graph) {
    //! LayerNorm
    auto norm_out = add_opr<LayerNorm>(
            device, name + ".norm", OpIOs{input}, m_embd, true, bias, true, eps)[0];
    //! matmul
    auto matmul_out = add_opr<MatMulLast>(
            device, name + ".output", OpIOs{norm_out},
            std::vector<size_t>{vocab, embd})[0];
    set_output(matmul_out);
}

void HeadModule::execute(WorkSpace* workspace, uint32_t nr_past, bool is_prefill) {
    //! prefill is no need to execute
    if (!is_prefill) {
        for (auto opr : oprs()) {
            opr->pre_execute();
            opr->execute(workspace, nr_past);
            opr->end_execute();
        }
    }
}

EmbdModule::EmbdModule(
        Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t vocab,
        UserConfig model_config, Device* device, const std::string& name)
        : OprModuleBase(input, device, name), m_embd(embd), m_graph(graph) {
    auto embd_out = add_opr<Embedding>(
            OpIOs{input}, embd, vocab, model_config.compt_type, device,
            "tok_embeddings")[0];
    set_output(embd_out);
}

//! Graph
///////////////////////////////////////////////////////////////////////////

size_t Graph::get_workspace_in_byte() {
    size_t max_workspace = 0;
    for (size_t i = 0; i < m_modules.size(); i++) {
        m_modules[i]->deduce_output_shape();
        size_t workspace = m_modules[i]->get_workspace_in_byte();
        max_workspace = workspace > max_workspace ? workspace : max_workspace;
    }
    return max_workspace;
}

void Graph::execute(
        std::vector<int32_t> in_token, std::vector<float>& logist, uint32_t nr_past,
        bool prefill) {
    if (m_input->dims() == 0 || !same_input_shape(in_token)) {
        m_input->set_shape({in_token.size()}, DType::Int32);
        size_t len = get_workspace_in_byte();
        if (m_workspace->ptr() == nullptr) {
            auto data = m_device->allocate(len);
            m_workspace->set_memory(data, len);
        } else if (m_workspace->ptr() && len > m_workspace->length()) {
            m_device->free_device(m_workspace->ptr());
            auto data = m_device->allocate(len);
            m_workspace->set_memory(data, len);
        }
    }
    m_input->resume_user_count();
    m_input->prepare_data();
    m_device->host2device_copy(
            m_input->ptr(), in_token.data(), in_token.size() * sizeof(int32_t), true);
    INFER_ASSERT(
            m_output->length() == logist.size(),
            "output length is not match with logist size");
    for (size_t i = 0; i < m_modules.size(); i++) {
        m_modules[i]->execute(m_workspace.get(), nr_past, prefill);
    }
    if (!prefill) {
        m_device->device2host_copy(
                logist.data(), m_output->ptr(), logist.size() * sizeof(float), true);
    }
    m_device->sync();
    m_output->recall_data();
}
void Graph::reset_ctx() {
    for (size_t i = 0; i < m_modules.size(); i++) {
        m_modules[i]->reset_ctx();
    }
}

void Graph::collect_weights() {
    //! collect all the weights
    for (auto module : m_modules) {
        auto all_weights = module->get_all_weights();
        for (auto weight : all_weights) {
            std::string name = weight->name();
            INFER_ASSERT(m_weights_map.count(name) == 0, "dumplicated weight.");
            m_weights_map[name] = weight;
        }
    }
}

DType Graph::convert_dtype(int32_t type) {
    switch (type) {
        case 0:
            return DType::Float32;
        case 1:
            return DType::Float16;
        case 2:
            return DType::Int4;
        case 3:
            return DType::Uint4;
        case 4:
            return DType::Int8;
        default:
            INFER_ASSERT(0, "unsupported weight type");
    }
};

std::string Graph::get_weight_alias(const std::string& name) {
    std::regex reg_get("\\.(\\d+)\\.");
    std::smatch match;
    //! if find in map directly
    if (m_weights_name_aliases.find(name) != m_weights_name_aliases.end()) {
        return m_weights_name_aliases[name];
        //! if matmul "xxx.[layer_num].xxx"
    } else if (std::regex_search(name, match, reg_get)) {
        auto layer_num = match[1].str();
        std::regex reg_replace("\\.\\d+\\.");
        std::string reg_name = regex_replace(name, reg_replace, ".x.");
        //! if "aaa.x.bbbb" is found
        if (m_weights_name_aliases.find(reg_name) != m_weights_name_aliases.end()) {
            auto tmp_alias = m_weights_name_aliases[reg_name];
            //! replace "cccc.x.dddd" to "cccc.[layer_num].dddd"
            std::regex regx("\\.x\\.");
            return regex_replace(tmp_alias, regx, "." + layer_num + ".");
        } else {
            return name;
        }
        //! return origin
    } else {
        return name;
    }
}

Graph::~Graph() {
    if (m_workspace->ptr()) {
        m_device->free_device(m_workspace->ptr());
    }
}

bool Graph::same_input_shape(std::vector<int32_t> in_token) {
    INFER_ASSERT(m_input->dims() == 1, "input tensor should be one dim.");
    return m_input->shape()[0] == in_token.size();
}

void Graph::load(
        std::shared_ptr<InputFile> fin, LlmParams& param,
        std::shared_ptr<Vocab> vocab) {
    // verify the magic number wrote when model convert
    uint32_t magic;
    uint32_t version = 0;
    fin->read_raw((char*)&magic, sizeof(magic));
    INFER_ASSERT(magic == 0x123456, "model magic is not create!!!!");
    load_param(fin, param, vocab);

    construct_llm();
    collect_weights();

    set_weights_alias();
    size_t weight_length = 0;
    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ftype;
        if (fin->eof()) {
            break;
        }

        fin->read_raw(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
        fin->read_raw(reinterpret_cast<char*>(&length), sizeof(length));
        fin->read_raw(reinterpret_cast<char*>(&ftype), sizeof(ftype));

        if (fin->eof()) {
            break;
        }

        size_t nr_number = 1;
        int32_t shape[2] = {1, 1};
        for (int i = 0; i < n_dims; ++i) {
            fin->read_raw(reinterpret_cast<char*>(&shape[i]), sizeof(shape[i]));
            nr_number *= shape[i];
        }

        std::string name(length, 0);
        fin->read_raw(&name[0], length);
        auto alias_name = get_weight_alias(name);
        if (m_weights_map.count(alias_name) == 0) {
            INFER_LOG("skip weight %s\n", alias_name.c_str());
            auto dtype = convert_dtype(ftype);
            size_t length = nr_number * dtype_in_byte(dtype) / dtype_block_size(dtype);
            fin->skip(length);
            continue;
        }
        INFER_ASSERT(
                m_weights_map.count(alias_name) == 1,
                "Error weight is not found when loading.");
        auto weight = m_weights_map[alias_name];
        if (weight->length() != nr_number) {
            INFER_LOG("weight %s is not match.\n", alias_name.c_str());
        }
        INFER_ASSERT(
                weight->length() == nr_number, "Error length of weight is mismatch.");
        weight->set_file(fin, fin->tell());
        weight->set_dtype(convert_dtype(ftype));
        fin->skip(weight->length_in_byte());
        weight_length += weight->length_in_byte();
    }
    INFER_LOG("total weight length = %lu\n", weight_length);
}
