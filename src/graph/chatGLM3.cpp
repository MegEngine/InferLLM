#include "chatGLM.h"

using namespace inferllm;

void ChatGLMGraph3::set_weights_alias() {
    m_weights_name_aliases.clear();
    // clang-format off
    m_weights_name_aliases = {
            {"transformer.embedding.word_embeddings.weight", "tok_embeddings.weight"},
            {"transformer.encoder.layers.x.input_layernorm.weight", "layers.x.attention_norm.weight"},
            {"transformer.encoder.layers.x.self_attention.query_key_value.weight", "layers.x.attention.wqkv.weight"},
            {"transformer.encoder.layers.x.self_attention.query_key_value.bias", "layers.x.attention.wqkv.bias"},
            {"transformer.encoder.layers.x.self_attention.dense.weight", "layers.x.attention.wo.weight"},
            {"transformer.encoder.layers.x.post_attention_layernorm.weight", "layers.x.ffn_norm.weight"},
            {"transformer.encoder.layers.x.mlp.dense_h_to_4h.weight", "layers.x.ffn.matmul1.weight"},
            {"transformer.encoder.layers.x.mlp.dense_4h_to_h.weight", "layers.x.ffn.matmul2.weight"},
            {"transformer.encoder.final_layernorm.weight", "head.norm.weight"},
            {"transformer.output_layer.weight", "head.output.weight"},
    };
    // clang-format on
}

//! LlamaGraph
void ChatGLMGraph3::load_param(
        std::shared_ptr<InputFile> fin, LlmParams& param,
        std::shared_ptr<Vocab> vocab) {
    Header header;
    // load model header
    fin->read_raw((char*)&header.param_offset, sizeof(header.param_offset));
    fin->read_raw((char*)&header.param_length, sizeof(header.param_length));
    fin->read_raw((char*)&header.vocab_offset, sizeof(header.vocab_offset));
    fin->read_raw((char*)&header.vocab_length, sizeof(header.vocab_length));
    fin->read_raw((char*)&header.tensor_offset, sizeof(header.tensor_offset));

    fin->seek(header.param_offset);
    // load param
    fin->read_raw((char*)&param.n_embd, sizeof(param.n_embd));
    fin->read_raw((char*)&param.n_head, sizeof(param.n_head));
    fin->read_raw((char*)&param.n_layer, sizeof(param.n_layer));
    fin->read_raw((char*)&param.n_mult, sizeof(param.n_mult));
    fin->read_raw((char*)&param.n_vocab, sizeof(param.n_vocab));
    int32_t multi_query;
    fin->read_raw((char*)&multi_query, sizeof(multi_query));
    param.is_multi_query = multi_query > 0;
    fin->read_raw((char*)&param.multi_query_group_num, sizeof(param.multi_query_group_num));

    m_param = param;
    

    // load vocab
    fin->seek(header.vocab_offset);
    vocab->load_vocab(fin, param.n_vocab);

    m_param.n_vocab = 65024;
    param.n_vocab = 65024;
    fin->seek(header.tensor_offset);
}

void ChatGLMGraph3::construct_llm() {
    m_input = std::make_shared<Tensor>(device(), name() + ":input");
    std::shared_ptr<Tensor> input = m_input;
    //! embd
    input = add_module<EmbdModule>(
            this, input, m_param.n_embd, m_param.n_vocab, model_config(), device(), "");

    int nr_layer = m_param.n_layer;
    for (int i = 0; i < nr_layer; i++) {
        std::string name = "layers." + std::to_string(i);
        //! layer norm
        std::shared_ptr<Tensor> attention_input = input;
        auto norm_out_attention =
                add_one_opr_module<LayerNorm>(
                        this, OpIOs{attention_input}, device(),
                        name + ".attention_norm")
                        ->add_opr(
                                m_param.n_embd, /*mul*/ true, /*bias*/ false,
                                /*rms*/ true);
        //! attentin
        auto attention_output = add_module<AttentionModule<Glm2MultiQueryAttention>>(
                this, norm_out_attention, m_param.n_embd, m_param.n_head,
                m_param.multi_query_group_num, m_param.n_ctx, model_config(), device(),
                name + ".attention", i, true /*fused_weights*/, true /*bias*/,
                RotMode::Mode0, false /*proj_bias*/);

        //! add  norm_out_attention * scale + attention_output
        auto add_output = add_one_opr_module<Elemwise>(
                                  this, OpIOs{attention_input, attention_output},
                                  device(), name + ".attention.Elemwise")
                                  ->add_opr(ElemMode::Add);

        std::shared_ptr<Tensor> feed_forward_input = add_output;
        //! layer normal
        auto ffn_norm_out =
                add_one_opr_module<LayerNorm>(
                        this, OpIOs{feed_forward_input}, device(), name + ".ffn_norm")
                        ->add_opr(
                                m_param.n_embd, /*mul*/ true, /*bias*/ false,
                                /*rms*/ true);
        //! feed forward
        auto ffn_output = add_module<Glm2FFNModule>(
                this, ffn_norm_out, m_param.n_embd, m_param.n_mult, model_config(),
                device(), name);
        //! add ffn_norm_out * scale + ffn_output
        input = add_one_opr_module<Elemwise>(
                        this, OpIOs{feed_forward_input, ffn_output}, device(),
                        name + ".ffn.Elemwise")
                        ->add_opr(ElemMode::Add);
    }
    //! the last layer
    m_output = add_module<HeadModule>(
            this, input, m_param.n_embd, m_param.n_vocab, model_config(), device(),
            "head");
}

void ChatGLMGraph3::post_tokenize(std::vector<Vocab::Id>& input) {
    std::vector<Vocab::Id> res;
    res.push_back(64790);
    res.push_back(64792);
    // add a space in the head
    input.insert(input.begin(), res.begin(), res.end());
}
