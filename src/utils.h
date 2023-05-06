// Various helper functions and utilities

#pragma once

#include <fstream>
#include <list>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include "file.h"

namespace inferllm {

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

//
// Vocab utils
//

//! the tokenizer vocabulary
class Vocab {
public:
    using Id = int32_t;
    using Token = std::string;
    struct TokenScore {
        Token tok;
        float score;
    };

    void load_vocab(std::shared_ptr<InputFile> fs, size_t size) {
        id_to_token.resize(size);
        std::string word;
        for (size_t i = 0; i < size; i++) {
            float score = 0;
            uint32_t len;
            fs->read_raw((char*)&len, sizeof(len));
            word.resize(len);
            fs->read_raw((char*)word.data(), len);

            token_to_id[word] = i;
            id_to_token[i].tok = word;
            id_to_token[i].score = score;
        }
    }

    void load_vocab_with_score(std::shared_ptr<InputFile> fs, size_t size) {
        id_to_token.resize(size);
        std::string word;
        for (size_t i = 0; i < size; i++) {
            float score = 0;
            uint32_t len;
            fs->read_raw((char*)&len, sizeof(len));
            word.resize(len);
            fs->read_raw((char*)word.data(), len);
            fs->read_raw((char*)&score, sizeof(score));

            token_to_id[word] = i;
            id_to_token[i].tok = word;
            id_to_token[i].score = score;
        }
    }

    Id map_to_id(const Token& str) { return token_to_id[str]; }

    Token unmap_to_token(Id id) { return id_to_token[id].tok; }

    std::map<Token, Id> token_to_id;
    std::vector<TokenScore> id_to_token;
};

// sample next token given probabilities for each embedding
//
//   - consider only the top K tokens
//   - from them, consider only the top tokens with cumulative probability > P
//
Vocab::Id llama_sample_top_p_top_k(const Vocab& vocab, const float* logits,
                                   std::list<Vocab::Id>& last_n_tokens,
                                   double repeat_penalty, int top_k,
                                   double top_p, double temp,
                                   std::mt19937& rng);

// filer to top K tokens from list of logits
void sample_top_k(std::vector<std::pair<double, Vocab::Id>>& logits_id,
                  int top_k);

std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

}  // namespace inferllm

//! branch prediction hint: likely to take
#define infer_likely(v) __builtin_expect(static_cast<bool>(v), 1)

//! branch prediction hint: unlikely to take
#define infer_unlikely(v) __builtin_expect(static_cast<bool>(v), 0)

#define INFER_LOG(format, ...) fprintf(stderr, format, ##__VA_ARGS__)
#define INFER_ERROR(format, ...) fprintf(stderr, format, ##__VA_ARGS__)

#define INFER_ASSERT(expr, message)                                           \
    do {                                                                      \
        if (infer_unlikely(!(expr))) {                                        \
            INFER_ERROR(                                                      \
                    "Assert \' %s \' failed at file : %s \n"                  \
                    "line %d : %s,\nextra "                                   \
                    "message: %s",                                            \
                    #expr, __FILE__, __LINE__, __PRETTY_FUNCTION__, message); \
            abort();                                                          \
        }                                                                     \
    } while (0)
