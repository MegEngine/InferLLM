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
 
#ifndef SRC_CORE_MODEL_IMP_H_
#define SRC_CORE_MODEL_IMP_H_

#include <list>
#include <map>
#include <memory>
#include <string>

#include "device.h"
#include "graph.h"
#include "kern/kernel_define.h"
#include "model.h"

namespace llm_learning {

//! the implement of model
class ModelImp {
 public:
  ModelImp(const ModelConfig& config, const std::string& name);
  //! load the model from model_path
  void load(const std::string& model_path);

  //! allocate memory for the model or init its param
  void init(uint32_t top_k, float top_p, float temp, float repeat_penalty, int repeat_last_n,
            int32_t seed) {
    m_top_k = top_k;
    m_top_p = top_p;
    m_temp = temp;
    m_repeat_penalty = repeat_penalty;
    m_repeat_last_n = repeat_last_n;
    for (uint32_t i = 0; i < m_repeat_last_n; i++) {
      m_last_queue.push_back(0);
    }
    m_rng = std::mt19937(seed);
  }

  //! prefill the model with inference with the given promote
  void prefill(const std::string& promote);

  //! decode the user input sentence
  std::string decode(const std::string& user_input, int& token);

  std::string decode_iter(int& token);

  uint32_t get_remain_token() { return m_graph->get_nr_ctx() - m_past; }

  void reset_token() {
    m_past = 0;
    m_graph->reset_ctx();
  }

  int32_t sample_and_update();

 private:
  std::vector<Vocab::Id> tokenize(const std::string& text, bool bos);

  uint32_t m_past = 0;

  uint32_t m_top_k;
  float m_top_p;
  float m_temp;
  float m_repeat_penalty;
  uint32_t m_repeat_last_n;

  int32_t m_pre_token;
  ModelType m_model_type;

  std::string m_name;
  LlmParams m_param;
  ModelConfig m_config;
  //! TODO: support run model on multi device later.
  std::unique_ptr<Device> m_device;
  std::shared_ptr<Graph> m_graph;
  std::shared_ptr<Vocab> m_vocab;
  std::list<int32_t> m_last_queue;
  std::vector<float> m_logist;

  std::mt19937 m_rng;
};

}  // namespace llm_learning

#endif  // SRC_CORE_MODEL_IMP_H_
