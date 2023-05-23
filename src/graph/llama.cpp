#include "llama.h"

using namespace inferllm;

void LlamaGraph::set_weights_alias() {
    m_weights_name_aliases.clear();
    m_weights_name_aliases = {
            {"norm.weight", "head.norm.weight"},
            {"output.weight", "head.output.weight"},
            {"layers.x.feed_forward.w3.weight", "layers.x.ffn.w3.weight"},
            {"layers.x.feed_forward.w2.weight", "layers.x.ffn.w2.weight"},
            {"layers.x.feed_forward.w1.weight", "layers.x.ffn.w1.weight"},
            {"layers.x.ffn_norm.weight", "layers.x.ffn.norm.weight"},
            {"layers.x.attention_norm.weight",
             "layers.x.attention.norm.weight"},
    };
}

//! LlamaGraph
void LlamaGraph::load(std::shared_ptr<InputFile> fin, LlmParams& param,
                      std::shared_ptr<Vocab> vocab) {
    // verify the magic number wrote when model convert
    uint32_t magic;
    uint32_t version = 0;
    fin->read_raw((char*)&magic, sizeof(magic));
    if (magic != 'ggml') {
        fin->read_raw((char*)&version, sizeof(version));
    }
    LlamaModelType model_type;
    if (magic == 'ggml' && version == 0) {
        model_type = LlamaModelType::LLAMA_FILE_VERSION_GGML;
    } else if (magic == 'ggmf' && version == 1) {
        model_type = LlamaModelType::LLAMA_FILE_VERSION_GGMF_V1;
    } else if (magic == 'ggjt' && version == 1) {
        model_type = LlamaModelType::LLAMA_FILE_VERSION_GGJT_V1;
    } else {
        INFER_ASSERT(0, "unsupported model type.");
    }

    INFER_LOG("model is %s , version = %d\n", magic != 'ggml' ? "new" : "old",
              version);

    // load param
    fin->read_raw((char*)&param.n_vocab, sizeof(param.n_vocab));
    fin->read_raw((char*)&param.n_embd, sizeof(param.n_embd));
    fin->read_raw((char*)&param.n_mult, sizeof(param.n_mult));
    fin->read_raw((char*)&param.n_head, sizeof(param.n_head));
    fin->read_raw((char*)&param.n_layer, sizeof(param.n_layer));
    fin->read_raw((char*)&param.n_rot, sizeof(param.n_rot));
    fin->read_raw((char*)&param.ftype, sizeof(param.ftype));
    param.n_ctx = param.n_ctx;

    INFER_LOG("%s: n_vocab         = %u\n", __func__, param.n_vocab);
    INFER_LOG("%s: n_ctx           = %u\n", __func__, param.n_ctx);
    INFER_LOG("%s: n_embd          = %u\n", __func__, param.n_embd);
    INFER_LOG("%s: n_mult          = %u\n", __func__, param.n_mult);
    INFER_LOG("%s: n_head          = %u\n", __func__, param.n_head);
    INFER_LOG("%s: n_layer         = %u\n", __func__, param.n_layer);
    INFER_LOG("%s: n_rot           = %u\n", __func__, param.n_rot);
    INFER_LOG("%s: model ftype     = %u\n", __func__, param.ftype);

    //! load vocabulary
    if (model_type == LlamaModelType::LLAMA_FILE_VERSION_GGJT_V1) {
        vocab->load_vocab_with_score(fin, param.n_vocab);
    } else {
        vocab->load_vocab(fin, param.n_vocab);
    }

    m_param = param;

    //! construct the llm graph with the input parameters
    constuct_llm();
    //! collect all the weights from the llm graph into the weights map
    collect_weights();

    //! read or mmap all the weights from the model file
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
        INFER_ASSERT(m_weights_map.count(alias_name) == 1,
                     "Error weight is not found when loading.");
        if (model_type >= LlamaModelType::LLAMA_FILE_VERSION_GGJT_V1) {
            // skip to the next multiple of 32 bytes
            fin->skip(-fin->tell() & 31);
        }
        auto weight = m_weights_map[alias_name];
        INFER_ASSERT(weight->length() == nr_number,
                     "Error length of weight is mismatch.");
        weight->set_file(fin, fin->tell());
        weight->set_dtype(convert_dtype(ftype));
        fin->skip(weight->length_in_byte());
        weight_length += weight->length_in_byte();
    }
    INFER_LOG("total weight length = %lu\n", weight_length);
}

void LlamaGraph::constuct_llm() {
    uint32_t embd = m_param.n_embd;
    uint32_t mult = m_param.n_mult;
    uint32_t head = m_param.n_head;
    uint32_t rot = m_param.n_rot;
    uint32_t ctx = m_param.n_ctx;
    uint32_t n_vocab = m_param.n_vocab;

    size_t nff = ((2 * (4 * embd) / 3 + mult - 1) / mult) * mult;
    m_input = std::make_shared<Tensor>(device(), name() + ":input");
    std::shared_ptr<Tensor> input = m_input;
    //! embd
    input = add_module<EmbdModule>(this, input, embd, n_vocab, model_config(),
                                   device(), "");

    int nr_layer = m_param.n_layer;
    for (int i = 0; i < nr_layer; i++) {
        std::string name = "layers." + std::to_string(i);
        //! layer norm
        std::shared_ptr<Tensor> attention_input = input;
        auto norm_out_attention = add_one_opr_module<LayerNorm>(
                                          this, OpIOs{attention_input},
                                          device(), name + ".attention.norm")
                                          ->add_opr(embd);
        //! attentin
        auto attention_output = add_module<AttentionModule<LlamaAttention>>(
                this, norm_out_attention, embd, head, rot, ctx, model_config(),
                device(), name + ".attention", i);
        //! add
        auto add_output =
                add_one_opr_module<Elemwise>(
                        this, OpIOs{attention_input, attention_output},
                        device(), name + ".attention_add")
                        ->add_opr(ElemMode::Add);

        std::shared_ptr<Tensor> feed_forward_input = add_output;
        //! layer normal
        auto ffn_norm_out =
                add_one_opr_module<LayerNorm>(this, OpIOs{feed_forward_input},
                                              device(), name + ".ffn.norm")
                        ->add_opr(embd);
        //! feed forward
        auto ffn_output = add_module<LlamaFFNModule>(
                this, ffn_norm_out, embd, nff, model_config(), device(), name);
        //! add
        input = add_one_opr_module<Elemwise>(
                        this, OpIOs{feed_forward_input, ffn_output}, device(),
                        name + ".ffn_add")
                        ->add_opr(ElemMode::Add);
    }
    //! the last layer
    m_output = add_module<HeadModule>(this, input, embd, n_vocab,
                                      model_config(), device(), "head");
}
