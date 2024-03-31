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
 
#include "utils.h"

#include <math.h>
#include <stdarg.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>

namespace llm_learning {

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>  // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__)
#include <alloca.h>
#endif

bool gpt_params_parse(int argc, char** argv, gpt_params& params) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-s" || arg == "--seed") {
      params.seed = std::stoi(argv[++i]);
    } else if (arg == "-t" || arg == "--threads") {
      params.n_threads = std::stoi(argv[++i]);
    } else if (arg == "-p" || arg == "--prompt") {
      params.interactive = false;
      params.interactive_start = false;
      params.use_color = false;

      params.prompt = argv[++i];
    } else if (arg == "-f" || arg == "--file") {
      params.interactive = false;
      params.interactive_start = false;
      params.use_color = false;

      std::ifstream file(argv[++i]);

      std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(),
                back_inserter(params.prompt));

    } else if (arg == "-n" || arg == "--n_predict") {
      params.n_predict = std::stoi(argv[++i]);
    } else if (arg == "--top_k") {
      params.top_k = std::stoi(argv[++i]);
    } else if (arg == "-c" || arg == "--ctx_size") {
      params.n_ctx = std::stoi(argv[++i]);
    } else if (arg == "--top_p") {
      params.top_p = std::stof(argv[++i]);
    } else if (arg == "--temp") {
      params.temp = std::stof(argv[++i]);
    } else if (arg == "--repeat_last_n") {
      params.repeat_last_n = std::stoi(argv[++i]);
    } else if (arg == "--repeat_penalty") {
      params.repeat_penalty = std::stof(argv[++i]);
    } else if (arg == "-b" || arg == "--batch_size") {
      params.n_batch = std::stoi(argv[++i]);
    } else if (arg == "-m" || arg == "--model") {
      params.model = argv[++i];
    } else if (arg == "-i" || arg == "--interactive") {
      params.interactive = true;
    } else if (arg == "--interactive-start") {
      params.interactive = true;
      params.interactive_start = true;
    } else if (arg == "--color") {
      params.use_color = true;
    } else if (arg == "-r" || arg == "--reverse-prompt") {
      params.antiprompt = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      gpt_print_usage(argc, argv, params);
      exit(0);
    } else {
      fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
      gpt_print_usage(argc, argv, params);
      exit(0);
    }
  }

  return true;
}

void gpt_print_usage(int argc, char** argv, const gpt_params& params) {
  fprintf(stderr, "usage: %s [options]\n", argv[0]);
  fprintf(stderr, "\n");
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -h, --help            show this help message and exit\n");
  fprintf(stderr, "  -i, --interactive     run in interactive mode\n");
  fprintf(stderr,
          "  --interactive-start   run in interactive mode and poll user input at startup\n");
  fprintf(stderr, "  -r PROMPT, --reverse-prompt PROMPT\n");
  fprintf(stderr,
          "                        in interactive mode, poll user input upon seeing PROMPT\n");
  fprintf(stderr,
          "  --color               colorise output to distinguish prompt and user input from "
          "generations\n");
  fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
  fprintf(stderr,
          "  -t N, --threads N     number of threads to use during computation (default: %d)\n",
          params.n_threads);
  fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
  fprintf(stderr, "                        prompt to start generation with (default: random)\n");
  fprintf(stderr, "  -f FNAME, --file FNAME\n");
  fprintf(stderr, "                        prompt file to start generation.\n");
  fprintf(stderr, "  -n N, --n_predict N   number of tokens to predict (default: %d)\n",
          params.n_predict);
  fprintf(stderr, "  --top_k N             top-k sampling (default: %d)\n", params.top_k);
  fprintf(stderr, "  --top_p N             top-p sampling (default: %.1f)\n", params.top_p);
  fprintf(stderr, "  --repeat_last_n N     last n tokens to consider for penalize (default: %d)\n",
          params.repeat_last_n);
  fprintf(stderr, "  --repeat_penalty N    penalize repeat sequence of tokens (default: %.1f)\n",
          params.repeat_penalty);
  fprintf(stderr, "  -c N, --ctx_size N    size of the prompt context (default: %d)\n",
          params.n_ctx);
  fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", params.temp);
  fprintf(stderr, "  -b N, --batch_size N  batch size for prompt processing (default: %d)\n",
          params.n_batch);
  fprintf(stderr, "  -m FNAME, --model FNAME\n");
  fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
  fprintf(stderr, "\n");
}

std::string gpt_random_prompt(std::mt19937& rng) {
  const int r = rng() % 10;
  switch (r) {
    case 0:
      return "So";
    case 1:
      return "Once upon a time";
    case 2:
      return "When";
    case 3:
      return "The";
    case 4:
      return "After";
    case 5:
      return "If";
    case 6:
      return "import";
    case 7:
      return "He";
    case 8:
      return "She";
    case 9:
      return "They";
    default:
      return "To";
  }

  return "The";
}

void replace(std::string& str, const std::string& needle, const std::string& replacement) {
  size_t pos = 0;
  while ((pos = str.find(needle, pos)) != std::string::npos) {
    str.replace(pos, needle.length(), replacement);
    pos += replacement.length();
  }
}

std::map<std::string, int32_t> json_parse(const std::string& fname) {
  std::map<std::string, int32_t> result;

  // read file into string
  std::string json;
  {
    std::ifstream ifs(fname);
    if (!ifs) {
      fprintf(stderr, "Failed to open %s\n", fname.c_str());
      exit(1);
    }

    json = std::string((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
  }

  if (json[0] != '{') {
    return result;
  }

  // parse json
  {
    bool has_key = false;
    bool in_token = false;

    std::string str_key = "";
    std::string str_val = "";

    int n = json.size();
    for (int i = 1; i < n; ++i) {
      if (!in_token) {
        if (json[i] == ' ') continue;
        if (json[i] == '"') {
          in_token = true;
          continue;
        }
      } else {
        if (json[i] == '\\' && i + 1 < n) {
          if (has_key == false) {
            str_key += json[i];
          } else {
            str_val += json[i];
          }
          ++i;
        } else if (json[i] == '"') {
          if (has_key == false) {
            has_key = true;
            ++i;
            while (json[i] == ' ') ++i;
            ++i;  // :
            while (json[i] == ' ') ++i;
            if (json[i] != '\"') {
              while (json[i] != ',' && json[i] != '}') {
                str_val += json[i++];
              }
              has_key = false;
            } else {
              in_token = true;
              continue;
            }
          } else {
            has_key = false;
          }

          replace(str_key, "\\u0120", " ");   // \u0120 -> space
          replace(str_key, "\\u010a", "\n");  // \u010a -> new line
          replace(str_key, "\\\"", "\"");     // \\\"   -> "

          try {
            result[str_key] = std::stoi(str_val);
          } catch (...) {
            // fprintf(stderr, "%s: ignoring key '%s' with value '%s'\n", fname.c_str(),
            // str_key.c_str(), str_val.c_str());
          }
          str_key = "";
          str_val = "";
          in_token = false;
          continue;
        }
        if (has_key == false) {
          str_key += json[i];
        } else {
          str_val += json[i];
        }
      }
    }
  }

  return result;
}

std::vector<Vocab::Id> gpt_tokenize(const Vocab& vocab, const std::string& text) {
  std::vector<std::string> words;

  // first split the text into words
  {
    std::string str = text;
    std::string pat =
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

    std::regex re(pat);
    std::smatch m;

    while (std::regex_search(str, m, re)) {
      for (auto x : m) {
        words.push_back(x);
      }
      str = m.suffix();
    }
  }

  // find the longest tokens that form the words:
  std::vector<Vocab::Id> tokens;
  for (const auto& word : words) {
    if (word.size() == 0) continue;

    int i = 0;
    int n = word.size();
    while (i < n) {
      int j = n;
      while (j > i) {
        auto it = vocab.token_to_id.find(word.substr(i, j - i));
        if (it != vocab.token_to_id.end()) {
          tokens.push_back(it->second);
          i = j;
          break;
        }
        --j;
      }
      if (i == n) {
        break;
      }
      if (j == i) {
        auto sub = word.substr(i, 1);
        if (vocab.token_to_id.find(sub) != vocab.token_to_id.end()) {
          tokens.push_back(vocab.token_to_id.at(sub));
        } else {
          fprintf(stderr, "%s: unknown token '%s'\n", __func__, sub.data());
        }
        ++i;
      }
    }
  }

  return tokens;
}

// TODO: Calculate this constant from the vocabulary
#define MAX_TOKEN_LEN 18
// SentencePiece implementation after https://guillaume-be.github.io/2020-05-30/sentence_piece
std::vector<Vocab::Id> llama_tokenize(const Vocab& vocab, const std::string& text, bool bos) {
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
      auto token = vocab.token_to_id.find(sub);
      if (token != vocab.token_to_id.end()) {
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
    auto token = vocab.id_to_token[token_id].tok;
    i -= token.length();
  }

  if (bos) {
    res.push_back(1);  // TODO: replace with vocab.bos
  }

  // Pieces are in reverse order so correct that
  std::reverse(res.begin(), res.end());

  return res;
}

void sample_top_k(std::vector<std::pair<double, Vocab::Id>>& logits_id, int top_k) {
  // find the top K tokens
  std::partial_sort(logits_id.begin(), logits_id.begin() + top_k, logits_id.end(),
                    [](const std::pair<double, Vocab::Id>& a,
                       const std::pair<double, Vocab::Id>& b) { return a.first > b.first; });

  logits_id.resize(top_k);
}

Vocab::Id llama_sample_top_p_top_k(const Vocab& vocab, const float* logits,
                                   std::list<Vocab::Id>& last_n_tokens, double repeat_penalty,
                                   int top_k, double top_p, double temp, std::mt19937& rng) {
  int n_logits = vocab.id_to_token.size();

  std::vector<std::pair<double, Vocab::Id>> logits_id;
  logits_id.reserve(n_logits);

  {
    const double scale = 1.0 / temp;
    for (int i = 0; i < n_logits; ++i) {
      // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
      // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
      if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
        // if score < 0 then repetition penalty has to multiplied to reduce the previous token
        // probability
        if (logits[i] < 0.0) {
          logits_id.push_back(std::make_pair(logits[i] * scale * repeat_penalty, i));
        } else {
          logits_id.push_back(std::make_pair(logits[i] * scale / repeat_penalty, i));
        }
      } else {
        logits_id.push_back(std::make_pair(logits[i] * scale, i));
      }
    }
  }

  sample_top_k(logits_id, top_k);

  double maxl = -INFINITY;
  for (const auto& kv : logits_id) {
    maxl = std::max(maxl, kv.first);
  }

  // compute probs for the top K tokens
  std::vector<double> probs;
  probs.reserve(logits_id.size());

  double sum = 0.0;
  for (const auto& kv : logits_id) {
    double p = exp(kv.first - maxl);
    probs.push_back(p);
    sum += p;
  }

  // normalize the probs
  for (auto& p : probs) {
    p /= sum;
  }

  if (top_p < 1.0f) {
    double cumsum = 0.0f;
    for (int i = 0; i < (int)probs.size(); i++) {
      cumsum += probs[i];
      if (cumsum >= top_p) {
        probs.resize(i + 1);
        logits_id.resize(i + 1);
        break;
      }
    }

    cumsum = 1.0 / cumsum;
    for (int i = 0; i < (int)probs.size(); i++) {
      probs[i] *= cumsum;
    }
  }

  // printf("\n");
  // for (int i = 0; i < (int) 10; i++) {
  //     printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
  // }
  // printf("\n\n");
  // exit(0);

  std::discrete_distribution<> dist(probs.begin(), probs.end());
  int idx = dist(rng);

  return logits_id[idx].second;
}

std::string format(const char* fmt, ...) {
  va_list ap, ap2;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  int size = vsnprintf(NULL, 0, fmt, ap);
  std::vector<char> buf(size + 1);
  int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
  va_end(ap2);
  va_end(ap);
  return std::string(buf.data(), size);
}

}  // namespace llm_learning
