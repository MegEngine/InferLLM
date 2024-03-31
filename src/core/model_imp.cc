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
 
#include "model_imp.h"

#include <algorithm>
#include <fstream>
#include <vector>

#include "file.h"
#include "graph.h"
#include "utils.h"

using namespace llm_learning;

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

ModelImp::ModelImp(const ModelConfig& config, const std::string& name)
    : m_name(name), m_config(config) {
  uint32_t nr_thread = config.nr_thread;
  m_vocab = std::make_shared<Vocab>();
#if INFER_X86
  m_device = std::make_unique<Device>(KernelType::X86, nr_thread);
#elif INFER_ARM
  m_device = std::make_unique<Device>(KernelType::Arm, nr_thread);
#else
  m_device = std::make_unique<Device>(KernelType::Naive, nr_thread);
#endif

  UserConfig user_config;
  user_config.compt_type = dtype_from_str(config.compt_type);
  user_config.weight_type = dtype_from_str(config.weight_type);
  m_graph = std::make_shared<Graph>(user_config, m_device.get(), name);
  m_past = 0;
}

void ModelImp::load(const std::string& model_path) {
  std::shared_ptr<InputFile> fin = std::make_shared<InputFile>(model_path, m_config.enable_mmap);

  // verify the magic number wrote when model convert
  uint32_t magic;
  uint32_t version = 0;
  fin->read_raw((char*)&magic, sizeof(magic));
  if (magic != 'ggml') {
    fin->read_raw((char*)&version, sizeof(version));
  }
  if (magic == 'ggml' && version == 0) {
    m_model_type = ModelType::LLAMA_FILE_VERSION_GGML;
  } else if (magic == 'ggmf' && version == 1) {
    m_model_type = ModelType::LLAMA_FILE_VERSION_GGMF_V1;
  } else if (magic == 'ggjt' && version == 1) {
    m_model_type = ModelType::LLAMA_FILE_VERSION_GGJT_V1;
  } else {
    INFER_ASSERT(0, "unsupported model type.");
  }

  INFER_LOG("model is %s , version = %d\n", magic != 'ggml' ? "new" : "old", version);

  // int n_parts = 1;
  //  load param
  auto& hparams = m_param;

  fin->read_raw((char*)&hparams.n_vocab, sizeof(hparams.n_vocab));
  fin->read_raw((char*)&hparams.n_embd, sizeof(hparams.n_embd));
  fin->read_raw((char*)&hparams.n_mult, sizeof(hparams.n_mult));
  fin->read_raw((char*)&hparams.n_head, sizeof(hparams.n_head));
  fin->read_raw((char*)&hparams.n_layer, sizeof(hparams.n_layer));
  fin->read_raw((char*)&hparams.n_rot, sizeof(hparams.n_rot));
  fin->read_raw((char*)&hparams.ftype, sizeof(hparams.ftype));

  INFER_LOG("%s: n_vocab         = %u\n", __func__, hparams.n_vocab);
  INFER_LOG("%s: n_ctx           = %u\n", __func__, hparams.n_ctx);
  INFER_LOG("%s: n_embd          = %u\n", __func__, hparams.n_embd);
  INFER_LOG("%s: n_mult          = %u\n", __func__, hparams.n_mult);
  INFER_LOG("%s: n_head          = %u\n", __func__, hparams.n_head);
  INFER_LOG("%s: n_layer         = %u\n", __func__, hparams.n_layer);
  INFER_LOG("%s: n_rot           = %u\n", __func__, hparams.n_rot);
  INFER_LOG("%s: model ftype     = %u\n", __func__, hparams.ftype);

  //! TODO: this param should pass when inference, now just write hard
  //! code
  hparams.n_ctx = 2048;
  m_logist.resize(m_param.n_vocab);

  //! load vocabulary
  if (m_model_type == ModelType::LLAMA_FILE_VERSION_GGJT_V1) {
    m_vocab->load_vocab_with_score(fin, m_param.n_vocab);
  } else {
    m_vocab->load_vocab(fin, m_param.n_vocab);
  }

  //! load the graph and weights
  //! TODO: support run model on multiply device
  m_graph->load(fin, m_param, m_model_type);
}

void ModelImp::prefill(const std::string& promote) {
  auto tokens = tokenize(promote, true);
  for (auto token : tokens) {
    m_last_queue.push_back(token);
    m_last_queue.pop_front();
  }
  m_graph->execute(tokens, m_logist, m_past, true);
  m_past = tokens.size();
}

//! decode the user input sentence
std::string ModelImp::decode(const std::string& user_input, int& token) {
  auto tokens = tokenize(user_input, false);
  for (auto token : tokens) {
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
  auto token = llama_sample_top_p_top_k(*m_vocab, m_logist.data(), m_last_queue, m_repeat_penalty,
                                        m_top_k, m_top_p, m_temp, m_rng);
  // update the last queue
  m_last_queue.push_back(token);
  m_last_queue.pop_front();
  m_pre_token = token;
  return token;
}

#define MAX_TOKEN_LEN 18
std::vector<Vocab::Id> ModelImp::tokenize(const std::string& text, bool bos) {
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
