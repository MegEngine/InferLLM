#pragma once

#include "tensor.h"
#include "kvstorage.h"
#include "kern/kernel.h"


namespace inferllm {

using OpIOs = std::vector<std::shared_ptr<Tensor>>;

//! Base class of an Op, the call step is:
//! call deduce_output_shape to get the output tensor shape
//! call init method to get init the op and compute the workspace
//! before execution, should call pre_execute to prepare the resource
//! call execute to get the compute result
//! call end execution to retrieve the resource
class OpBase {
public:
    OpBase(Device* device, const std::string& name, OpIOs inputs)
            : m_device(device), m_inputs(inputs), m_name(name) {
        for (auto input : m_inputs) {
            input->add_user();
        }
    }

    virtual void pre_execute() {
        for (auto weight : m_weights) {
            weight->prepare_data();
        }
        for (auto output : m_outputs) {
            if (output->get_curr_user_count() == 0 && !output->shared()) {
                output->resume_user_count();
                output->prepare_data();
            }
        }
    };

    virtual void execute(WorkSpace* workspace, uint32_t nr_past) {}

    virtual void end_execute() {
        for (auto input : m_inputs) {
            input->decrease_curr_user_count();
        }
    };

    virtual void deduce_output_shape() {
        m_outputs[0]->set_shape(m_inputs[0]->shape(), m_inputs[0]->dtype());
    };

    virtual size_t get_workspace_in_byte() {
        return 0;
    }

    virtual void load_weights(std::ifstream&){};

    virtual uint32_t nr_weights() { return 1; };

    //! init the op, and return the workspace need when execute
    virtual void init(OpIOs, OpIOs, WorkSpace*){};

    Device* device() { return m_device; }

    Kernel* get_kernel() { return m_device->kernel(); }

    void set_weights(OpIOs weights) { m_weights = weights; }
    void add_outputs(std::shared_ptr<Tensor> output) {
        output->set_owner_op(this);
        m_outputs.push_back(output);
    }
    void set_name(std::string name) { m_name = name; }

    OpIOs weights() { return m_weights; }
    OpIOs inputs() { return m_inputs; }
    OpIOs outputs() { return m_outputs; }
    std::string name() { return m_name; }

private:
    Device* m_device;
    OpIOs m_weights;
    OpIOs m_inputs;
    OpIOs m_outputs;
    std::string m_name;
};

class LayerNorm : public OpBase {
public:
    LayerNorm(Device* device, const std::string& name, OpIOs inputs,
              size_t embd, bool mul = true, bool bias = false, bool rms = true)
            : OpBase(device, name, inputs), m_mul(mul), m_bias(bias) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
        std::vector<std::shared_ptr<Tensor>> weights;
        if (m_mul) {
            weights.push_back(
                    std::make_shared<Tensor>(device, name + ".weight"));
            weights.back()->set_shape({embd}, DType::Float32);
        }
        if (m_bias) {
            weights.push_back(std::make_shared<Tensor>(device, name + ".bias"));
            weights.back()->set_shape({embd}, DType::Float32);
        }
        set_weights(weights);
    }

    void execute(WorkSpace* workspace, uint32_t nr_past) override;

private:
    bool m_mul;
    bool m_bias;
    bool m_rms;
};

class MatMul : public OpBase {
public:
    MatMul(Device* device, const std::string& name, OpIOs inputs,
           std::vector<size_t> shape, bool bias = false)
            : OpBase(device, name, inputs), m_bias(bias) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
        std::vector<std::shared_ptr<Tensor>> weights;
        auto weight = std::make_shared<Tensor>(device, name + ".weight");
        weight->set_shape(shape);
        weights.push_back(weight);
        if (bias) {
            auto bias = std::make_shared<Tensor>(device, name + ".bias");
            std::vector<size_t> bias_shape;
            bias_shape.push_back(shape[0]);
            bias->set_shape(bias_shape);
            weights.push_back(bias);
        }
        set_weights(weights);
    }

    virtual void deduce_output_shape() override {
        auto weight_shape = weights()[0]->shape();
        auto input_shape = inputs()[0]->shape();
        size_t M = input_shape.size() == 2 ? input_shape[0] : input_shape[1];
        size_t K = weight_shape[1];
        size_t N = weight_shape[0];
        outputs()[0]->set_shape({M, N}, inputs()[0]->dtype());
        // INFER_LOG("Matmul shape is M/K/N = %zu, %zu, %zu\n", M, K, N);
    }
    virtual void execute(WorkSpace* workspace, uint32_t nr_past) override;

    size_t get_workspace_in_byte() override;

    bool m_bias = false;
};

class MatMulLast : public MatMul {
public:
   using MatMul::MatMul; 

    void deduce_output_shape() override {
        auto weight_shape = weights()[0]->shape();
        auto input_shape = inputs()[0]->shape();
        //! only compute the last token
        size_t M = 1;
        size_t K = weight_shape[1];
        size_t N = weight_shape[0];
        outputs()[0]->set_shape({M, N}, inputs()[0]->dtype());
    }
    void execute(WorkSpace* workspace, uint32_t nr_past) override;

    size_t get_workspace_in_byte() override;
};

class SoftMax : public OpBase {
public:
    SoftMax(Device* device, const std::string& name, OpIOs inputs)
            : OpBase(device, name, inputs) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
    }
    void execute(WorkSpace* workspace, uint32_t nr_past) override;
};

class Reshape : public OpBase {
public:
    Reshape(Device* device, const std::string& name, OpIOs inputs,
            std::vector<int> shape)
            : OpBase(device, name, inputs), m_target_shape(shape) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
    }

    void deduce_output_shape() override {
        size_t len = inputs()[0]->length();
        std::vector<size_t> out_shape;
        out_shape.resize(m_target_shape.size());
        int count = 0;
        for (size_t i = 0; i < m_target_shape.size(); i++) {
            if (m_target_shape[i] != -1) {
                out_shape[i] = m_target_shape[i];
                INFER_ASSERT(len % m_target_shape[i] == 0, "Reshape error.\n");
                len = len / m_target_shape[i];
            } else {
                count++;
            }
        }
        INFER_ASSERT(count == 1, "multi -1 in Reshape param.\n");
        for (size_t i = 0; i < m_target_shape.size(); i++) {
            if (m_target_shape[i] == -1) {
                out_shape[i] = len;
            }
        }
        outputs()[0]->set_shape(out_shape, inputs()[0]->dtype());
    }

private:
    std::vector<int> m_target_shape;
};

class Elemwise : public OpBase {
public:
    Elemwise(Device* device, const std::string& name, OpIOs inputs,
             ElemMode mode, float scale = -INFINITY)
            : OpBase(device, name, inputs), m_mode(mode), m_scale(scale) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
    }
    void execute(WorkSpace* workspace, uint32_t nr_past) override;

private:
    float m_scale;
    ElemMode m_mode;
};

class DiagMask : public OpBase {
public:
    DiagMask(Device* device, const std::string& name, OpIOs inputs)
            : OpBase(device, name, inputs) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out0"));
    }
    void execute(WorkSpace* workspace, uint32_t nr_past) override;
};

//! Attention with cached the KvStorage, and output the kv with cache, and q
//! a*(wq, wk, qv) = q, k, v
//! out = softmax(q*k)*v
class LlamaAttention : public OpBase {
public:
    LlamaAttention(Device* device, const std::string& name, OpIOs inputs,
                   uint32_t embd, uint32_t rot, uint32_t nr_ctx, uint32_t head,
                   KvStorage* kstorage, KvStorage* vstorage, uint32_t layer_id,
                   bool fused_weights = false, bool bias = false,
                   bool rotary_weight = false)
            : OpBase(device, name, inputs),
              m_embd(embd),
              m_head(head),
              m_rot(rot),
              m_ctx(nr_ctx),
              m_fused_weights(fused_weights),
              m_bias(bias),
              m_rotary_weights(rotary_weight),
              m_kstorage(kstorage),
              m_vstorage(vstorage) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out"));
        std::vector<std::shared_ptr<Tensor>> weights;
        if (m_fused_weights) {
            auto weight_fused =
                    std::make_shared<Tensor>(device, name + ".wqkv.weight");
            weight_fused->set_shape(std::vector<size_t>{m_embd * 3, m_embd});
            weights.push_back(weight_fused);
            if (m_bias) {
                auto weight_bias =
                        std::make_shared<Tensor>(device, name + ".wqkv.bias");
                weight_bias->set_shape(std::vector<size_t>{m_embd * 3});
                weights.push_back(weight_bias);
            }
        } else {
            auto weight_q =
                    std::make_shared<Tensor>(device, name + ".wq.weight");
            weight_q->set_shape(std::vector<size_t>{embd, embd});
            auto weight_k =
                    std::make_shared<Tensor>(device, name + ".wk.weight");
            weight_k->set_shape(std::vector<size_t>{embd, embd});
            auto weight_v =
                    std::make_shared<Tensor>(device, name + ".wv.weight");
            weight_v->set_shape(std::vector<size_t>{embd, embd});
            weights.push_back(weight_q);
            weights.push_back(weight_k);
            weights.push_back(weight_v);
            if (m_bias) {
                auto bias_q =
                        std::make_shared<Tensor>(device, name + ".wq.bias");
                bias_q->set_shape(std::vector<size_t>{embd});
                auto bias_k =
                        std::make_shared<Tensor>(device, name + ".wk.bias");
                bias_k->set_shape(std::vector<size_t>{embd});
                auto bias_v =
                        std::make_shared<Tensor>(device, name + ".wv.bias");
                bias_v->set_shape(std::vector<size_t>{embd});
                weights.push_back(bias_q);
                weights.push_back(bias_k);
                weights.push_back(bias_v);
            }
        }
        if (m_rotary_weights) {
            auto rotary =
                    std::make_shared<Tensor>(device, name + ".rotary.inv_freq");
            rotary->set_shape(std::vector<size_t>{m_head});
            weights.push_back(rotary);
        }
        set_weights(weights);
    }

    void pre_execute() override{
        for (auto weight : weights()) {
            weight->prepare_data();
        }
        auto output = outputs()[0];
        if (output->get_curr_user_count() == 0) {
            output->prepare_data();
            output->resume_user_count();
        }
        m_kstorage->prepare_data();
        m_vstorage->prepare_data();
    }

    void execute(WorkSpace* workspace, uint32_t nr_past) override;

    void end_execute() override {
        for (auto weight : weights()) {
            weight->recall_data();
        }
        for (auto input : inputs()) {
            input->decrease_curr_user_count();
        }
        auto token_len = inputs()[0]->shape()[0];
        m_kstorage->add_id(token_len);
        m_vstorage->add_id(token_len);
        m_kstorage->recall_data();
        m_vstorage->recall_data();
    }

    size_t get_workspace_in_byte() override;

private:
    uint32_t m_embd;
    uint32_t m_head;
    uint32_t m_rot;
    uint32_t m_ctx;
    bool m_fused_weights;
    bool m_bias;
    bool m_rotary_weights;
    KvStorage* m_kstorage;
    KvStorage* m_vstorage;
};

class GlmAttention : public OpBase {
public:
    GlmAttention(Device* device, const std::string& name, OpIOs inputs,
                 uint32_t embd, uint32_t rot, uint32_t nr_ctx, uint32_t head,
                 KvStorage* kstorage, KvStorage* vstorage, uint32_t layer_id,
                 bool fused_weights = false, bool bias = false,
                 bool rotary_weight = false)
            : OpBase(device, name, inputs),
              m_embd(embd),
              m_head(head),
              m_rot(rot),
              m_ctx(nr_ctx),
              m_layer_id(layer_id),
              m_fused_weights(fused_weights),
              m_bias(bias),
              m_rotary_weights(rotary_weight),
              m_kstorage(kstorage),
              m_vstorage(vstorage) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out"));
        std::vector<std::shared_ptr<Tensor>> weights;
        if (m_fused_weights) {
            auto weight_fused =
                    std::make_shared<Tensor>(device, name + ".wqkv.weight");
            weight_fused->set_shape(std::vector<size_t>{m_embd * 3, m_embd});
            weights.push_back(weight_fused);
            if (m_bias) {
                auto weight_bias =
                        std::make_shared<Tensor>(device, name + ".wqkv.bias");
                weight_bias->set_shape(std::vector<size_t>{m_embd * 3});
                weights.push_back(weight_bias);
            }
        } else {
            auto weight_q =
                    std::make_shared<Tensor>(device, name + ".wq.weight");
            weight_q->set_shape(std::vector<size_t>{embd, embd});
            auto weight_k =
                    std::make_shared<Tensor>(device, name + ".wk.weight");
            weight_k->set_shape(std::vector<size_t>{embd, embd});
            auto weight_v =
                    std::make_shared<Tensor>(device, name + ".wv.weight");
            weight_v->set_shape(std::vector<size_t>{embd, embd});
            weights.push_back(weight_q);
            weights.push_back(weight_k);
            weights.push_back(weight_v);
            if (m_bias) {
                auto bias_q =
                        std::make_shared<Tensor>(device, name + ".wq.bias");
                bias_q->set_shape(std::vector<size_t>{embd});
                auto bias_k =
                        std::make_shared<Tensor>(device, name + ".wk.bias");
                bias_k->set_shape(std::vector<size_t>{embd});
                auto bias_v =
                        std::make_shared<Tensor>(device, name + ".wv.bias");
                bias_v->set_shape(std::vector<size_t>{embd});
                weights.push_back(bias_q);
                weights.push_back(bias_k);
                weights.push_back(bias_v);
            }
        }
        if (m_rotary_weights) {
            auto rotary =
                    std::make_shared<Tensor>(device, name + ".rotary.inv_freq");
            rotary->set_shape(std::vector<size_t>{m_head});
            weights.push_back(rotary);
        }
        set_weights(weights);
    }

    void pre_execute() override{
        for (auto weight : weights()) {
            weight->prepare_data();
        }
        auto output = outputs()[0];
        if (output->get_curr_user_count() == 0) {
            output->prepare_data();
            output->resume_user_count();
        }
        m_kstorage->prepare_data();
        m_vstorage->prepare_data();
    }

    void execute(WorkSpace* workspace, uint32_t nr_past) override;

    void end_execute() override {
        for (auto weight : weights()) {
            weight->recall_data();
        }
        for (auto input : inputs()) {
            input->decrease_curr_user_count();
        }
        auto token_len = inputs()[0]->shape()[0];
        m_kstorage->add_id(token_len);
        m_vstorage->add_id(token_len);
        m_kstorage->recall_data();
        m_vstorage->recall_data();
    }

    size_t get_workspace_in_byte() override;

private:
    uint32_t m_embd;
    uint32_t m_head;
    uint32_t m_rot;
    uint32_t m_ctx;
    uint32_t m_layer_id;
    uint32_t m_gmask_position;
    
    bool m_fused_weights;
    bool m_bias;
    bool m_rotary_weights;
    KvStorage* m_kstorage;
    KvStorage* m_vstorage;
};

class Embedding : public OpBase {
public:
    Embedding(OpIOs inputs, uint32_t embd, uint32_t vocab,
              DType compt_type, Device* device, const std::string& name)
            : OpBase(device, name, inputs),
              m_embd(embd),
              m_vocab(vocab),
              m_comp_type(compt_type) {
        add_outputs(std::make_shared<Tensor>(device, name + "_out0"));

        auto embeddings = std::make_shared<Tensor>(device, name + ".weight");
        std::vector<size_t> shape = {(size_t)vocab, (size_t)embd};
        embeddings->set_shape(shape);
        set_weights({embeddings});
    }
    void deduce_output_shape() override {
        size_t len = inputs()[0]->shape()[0];
        outputs()[0]->set_shape({len, m_embd}, m_comp_type);
    }

    void execute(WorkSpace* workspace, uint32_t nr_past) override;

private:
    DType m_comp_type;
    uint32_t m_embd;
    uint32_t m_vocab;
};

}  // namespace inferllm
