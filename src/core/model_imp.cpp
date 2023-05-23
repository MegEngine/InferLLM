#include <algorithm>
#include <fstream>
#include <vector>

#include "model_imp.h"
#include "graph.h"
#include "utils.h"
#include "file.h"

using namespace inferllm;

void ModelImp::load(const std::string& model_path){
    m_vocab = std::make_shared<Vocab>();
    std::shared_ptr<InputFile> fin =
            std::make_shared<InputFile>(model_path, m_config.enable_mmap);

    m_param.n_ctx = m_config.nr_ctx;
    m_graph->load(fin, m_param, m_vocab);
    m_logist.resize(m_param.n_vocab);
}

void ModelImp::prefill(const std::string& promote) {
    auto tokens = tokenize(promote, true);
    m_graph->post_tokenize(tokens);
    for(auto token : tokens) {
        m_last_queue.push_back(token);
        m_last_queue.pop_front();
    }
    m_graph->execute(tokens, m_logist, m_past, true);
    m_past = tokens.size();
}

//! decode the user input sentence
std::string ModelImp::decode(const std::string& user_input, int& token) {
    auto tokens = tokenize(user_input, false);
    m_graph->post_tokenize(tokens);
    for(auto token : tokens) {
        m_last_queue.push_back(token);
        m_last_queue.pop_front();
    }
    m_graph->execute(tokens, m_logist, m_past, false);
    sample_and_update();
    m_past += tokens.size();
    token = m_pre_token;
    return m_vocab->id_to_token[m_pre_token].tok;
}

//! decode the user input sentence
std::string ModelImp::decode_iter(int& token) {
    m_graph->execute({m_pre_token}, m_logist, m_past);
    sample_and_update();
    m_past++;
    token = m_pre_token;
    return m_vocab->id_to_token[m_pre_token].tok;
}

int32_t ModelImp::sample_and_update() {
    // sample the next token
    auto token = llama_sample_top_p_top_k(*m_vocab, m_logist.data(),
                                          m_last_queue, m_repeat_penalty,
                                          m_top_k, m_top_p, m_temp, m_rng);
    // update the last queue
    m_last_queue.push_back(token);
    m_last_queue.pop_front();
    m_pre_token = token;
    return token;
}

#define MAX_TOKEN_LEN 18
std::vector<Vocab::Id> ModelImp::tokenize(const std::string& text,
                                              bool bos) {
    std::vector<Vocab::Id> res;
    std::vector<int> score;
    std::vector<Vocab::Id> prev;
    int len = text.length();

    score.resize(len + 1);
    prev.resize(len + 1);

    // Forward pass
    for (int i = 0; i < len; i++) {
        int max_len = std::min(len - i, MAX_TOKEN_LEN);
        for (int sub_len = 1; sub_len <= len - i; sub_len++) {
            auto sub = text.substr(i, sub_len);
            auto token = m_vocab->token_to_id.find(sub);
            if (token != m_vocab->token_to_id.end()) {
                int token_score = sub.length() * sub.length();
                int local_score = score[i] + token_score;
                int next = i + sub_len;
                if (score[next] < local_score) {
                    score[next] = local_score;
                    prev[next] = (*token).second;
                }
            }
        }
    }

    // Backward pass
    int i = len;
    while (i > 0) {
        Vocab::Id token_id = prev[i];
        if (token_id == 0) {
            // TODO: Return error or something more meaningful
            printf("failed to tokenize string!\n");
            break;
        }
        res.push_back(token_id);
        auto token = m_vocab->id_to_token[token_id].tok;
        i -= token.length();
    }

    if (bos) {
        res.push_back(1);  // TODO: replace with vocab.bos
    }

    // Pieces are in reverse order so correct that
    std::reverse(res.begin(), res.end());

    return res;
}
