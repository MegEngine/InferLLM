/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Various helper functions and utilities

#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "file.h"

namespace llm_learning {
//
// CLI argument parsing
//

// The default parameters
struct gpt_params {
  int32_t seed = -1;  // RNG seed
  int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
  int32_t n_predict = 128;     // new tokens to predict
  int32_t repeat_last_n = 64;  // last n tokens to penalize
  int32_t n_ctx = 2048;        // context size

  // sampling parameters
  int32_t top_k = 40;
  float top_p = 0.95f;
  float temp = 0.10f;
  float repeat_penalty = 1.30f;

  int32_t n_batch = 8;  // batch size for prompt processing

  std::string model = "ggml-alpaca-7b-q4.bin";  // model path
  std::string prompt;

  bool use_color = true;  // use color to distinguish generations and inputs

  bool interactive = true;        // interactive mode
  bool interactive_start = true;  // reverse prompt immediately
  std::string antiprompt = "";    // string upon seeing which more user input is prompted
};

bool gpt_params_parse(int argc, char** argv, gpt_params& params);

void gpt_print_usage(int argc, char** argv, const gpt_params& params);

std::string gpt_random_prompt(std::mt19937& rng);

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

void replace(std::string& str, const std::string& needle, const std::string& replacement);

// poor-man's JSON parsing
std::map<std::string, int32_t> json_parse(const std::string& fname);

// split text into tokens
//
// ref:
// https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L53
//
// Regex (Python):
// r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
//
// Regex (C++):
// R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+|
// ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)"
//
std::vector<Vocab::Id> gpt_tokenize(const Vocab& vocab, const std::string& text);

// TODO: this is probably wrong, but I cannot figure out how this tokenizer
// works .. ref: https://github.com/google/sentencepiece
std::vector<Vocab::Id> llama_tokenize(const Vocab& vocab, const std::string& text, bool bos);

// sample next token given probabilities for each embedding
//
//   - consider only the top K tokens
//   - from them, consider only the top tokens with cumulative probability > P
//
Vocab::Id llama_sample_top_p_top_k(const Vocab& vocab, const float* logits,
                                   std::list<Vocab::Id>& last_n_tokens, double repeat_penalty,
                                   int top_k, double top_p, double temp, std::mt19937& rng);

// filer to top K tokens from list of logits
void sample_top_k(std::vector<std::pair<double, Vocab::Id>>& logits_id, int top_k);

std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

}  // namespace llm_learning

//! branch prediction hint: likely to take
#define infer_likely(v) __builtin_expect(static_cast<bool>(v), 1)

//! branch prediction hint: unlikely to take
#define infer_unlikely(v) __builtin_expect(static_cast<bool>(v), 0)

#define INFER_LOG(format, ...) fprintf(stderr, format, ##__VA_ARGS__)
#define INFER_ERROR(format, ...) fprintf(stderr, format, ##__VA_ARGS__)

#define INFER_ASSERT(expr, message)                                 \
  do {                                                              \
    if (infer_unlikely(!(expr))) {                                  \
      INFER_ERROR(                                                  \
          "Assert \' %s \' failed at file : %s \n"                  \
          "line %d : %s,\nextra "                                   \
          "message: %s",                                            \
          #expr, __FILE__, __LINE__, __PRETTY_FUNCTION__, message); \
      abort();                                                      \
    }                                                               \
  } while (0)

#endif