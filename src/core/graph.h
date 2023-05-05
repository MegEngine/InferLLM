#pragma once

#include <unordered_map>
#include "kvstorage.h"
#include "op.h"
#include "tensor.h"

namespace inferllm {

//! TODO: redefine this member of the struct
struct LlmParams {
    int32_t n_vocab;
    int32_t n_embd;
    int32_t n_mult;
    int32_t n_head;
    int32_t n_layer;
    int32_t n_rot;
    int32_t ftype;
    int32_t n_ctx;  // this is provided as user input?
};

struct UserConfig {
    DType compt_type;
};

class Graph;

class OprBlockBase {
public:
    OprBlockBase(std::shared_ptr<Tensor> input, Device* device,
                 const std::string& name);
    size_t get_workspace_in_byte();
    void deduce_output_shape();

    virtual void execute(WorkSpace* workspace, uint32_t nr_past,
                         bool is_prefill = false);

    template <typename Op, typename... Args>
    std::vector<std::shared_ptr<Tensor>> add_opr(Args&&... args) {
        auto opr = std::make_shared<Op>(std::forward<Args>(args)...);
        m_oprs.push_back(opr);
        return opr->outputs();
    }

    std::vector<std::shared_ptr<Tensor>> get_all_weights() {
        std::vector<std::shared_ptr<Tensor>> all_weights;
        for (auto opr : m_oprs) {
            auto weights = opr->weights();
            all_weights.insert(all_weights.end(), weights.begin(),
                               weights.end());
        }
        return all_weights;
    }

    std::shared_ptr<Tensor> input() const { return m_input; };
    std::shared_ptr<Tensor> output() const { return m_output; };

    void set_input(std::shared_ptr<Tensor> input) { m_input = input; };
    void set_output(std::shared_ptr<Tensor> output) { m_output = output; };

    std::string name() const { return m_name; }

    virtual void reset_ctx() {}

    std::vector<std::shared_ptr<OpBase>> oprs() { return m_oprs; }

private:
    std::string m_name;
    Device* m_device;
    std::shared_ptr<Tensor> m_input;
    std::shared_ptr<Tensor> m_output;
    std::vector<std::shared_ptr<OpBase>> m_oprs;
};

class AttentionBlock : public OprBlockBase {
public:
    AttentionBlock(Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd,
                   uint32_t head, uint32_t n_rot, uint32_t n_ctx,
                   UserConfig model_config, Device* device,
                   const std::string& name);

    void reset_ctx() override {
        m_kstorage->reset_id();
        m_vstorage->reset_id();
    }

private:
    uint32_t m_embd;
    uint32_t m_head;
    uint32_t m_rot;
    uint32_t m_index;

    Graph* m_graph;
    std::unique_ptr<KvStorage> m_kstorage;
    std::unique_ptr<KvStorage> m_vstorage;
};

class FeedForwardBlock : public OprBlockBase {
public:
    FeedForwardBlock(Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd,
                     uint32_t mult, UserConfig model_config,
                     Device* device, const std::string& name);

private:
    uint32_t m_embd;
    Graph* m_graph;
};

class HeadBlock : public OprBlockBase {
public:
    HeadBlock(Graph* graph, std::shared_ptr<Tensor> m_output, uint32_t embd,
              uint32_t vocab, UserConfig model_config, Device* device,
              const std::string& name);

    void execute(WorkSpace* workspace, uint32_t nr_past,
                 bool is_prefill = false) override;

private:
    uint32_t m_embd;
    uint32_t m_vocab;
    Graph* m_graph;
};

class EmbdBlock : public OprBlockBase {
public:
    EmbdBlock(Graph* graph, std::shared_ptr<Tensor> m_output, uint32_t embd,
              uint32_t vocab, UserConfig model_config, Device* device,
              const std::string& name);

private:
    uint32_t m_embd;
    uint32_t m_vocab;
    Graph* m_graph;
};

class Graph : public std::enable_shared_from_this<Graph> {
public:
    Graph(UserConfig model_config, Device* device, const std::string& name)
            : m_name(name), m_model_config(model_config), m_device(device) {
        m_workspace = make_unique<WorkSpace>();
    }

    static std::shared_ptr<Graph> make_graph(UserConfig model_config,
                                             Device* device,
                                             const std::string& name);

    virtual ~Graph();

    void execute(std::vector<int32_t> in_token, std::vector<float>& logist,
                 uint32_t nr_past, bool prefill = false);

    Device* device() { return m_device; }

    std::string name() { return m_name; }

    UserConfig model_config() { return m_model_config; }

    size_t get_workspace_in_byte();

    template <typename OpBlock, typename... Args>
    std::shared_ptr<Tensor> add_block(Args&&... args) {
        auto block = std::make_shared<OpBlock>(std::forward<Args>(args)...);
        m_blocks.push_back(block);
        return block->output();
    }

    void reset_ctx();

    bool same_input_shape(std::vector<int32_t> in_token);

    virtual void load(std::shared_ptr<InputFile> fin, LlmParams& param,
                      std::shared_ptr<Vocab> vocab) = 0;

    virtual uint32_t get_nr_ctx() = 0;

    virtual uint32_t get_nr_vocab() = 0;

    virtual void constuct_llm() = 0;

    std::shared_ptr<Tensor> m_input;
    std::shared_ptr<Tensor> m_output;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> m_weights_map;
    std::vector<std::shared_ptr<OprBlockBase>> m_blocks;

private:

    std::string m_name;
    UserConfig m_model_config;
    Device* m_device = nullptr;

    std::shared_ptr<Tensor> m_embeddings;
    std::unique_ptr<WorkSpace> m_workspace;
};
}  // namespace inferllm
