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
#ifndef SRC_CORE_GRAPH_H_
#define SRC_CORE_GRAPH_H_

#include <unordered_map>

#include "kvstorage.h"
#include "op.h"
#include "tensor.h"

namespace llm_learning {

//! TODO: redefine this member of the struct
struct LlmParams {
  int32_t n_vocab = 32000;
  int32_t n_ctx = 512;  // this is provided as user input?
  int32_t n_embd = 4096;
  int32_t n_mult = 256;
  int32_t n_head = 32;
  int32_t n_layer = 32;
  int32_t n_rot = 64;
  int32_t ftype = 0;
};

struct UserConfig {
  DType compt_type;
  DType weight_type;
};

enum class ModelType {
  LLAMA_FILE_VERSION_GGML = 0,
  LLAMA_FILE_VERSION_GGMF_V1,  // added version field and scores in vocab
  LLAMA_FILE_VERSION_GGJT_V1
};

class Graph;

class OprBlockBase {
 public:
  OprBlockBase(std::shared_ptr<Tensor> input, Device* device, const std::string& name);
  size_t get_workspace_in_byte();
  void deduce_output_shape();

  virtual void execute(WorkSpace* workspace, uint32_t nr_past, bool is_prefill = false);

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
      all_weights.insert(all_weights.end(), weights.begin(), weights.end());
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
  AttentionBlock(Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t head,
                 uint32_t n_rot, uint32_t n_ctx, UserConfig model_config, Device* device,
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
  FeedForwardBlock(Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t mult,
                   UserConfig model_config, Device* device, const std::string& name);

 private:
  uint32_t m_embd;
  Graph* m_graph;
};

class HeadBlock : public OprBlockBase {
 public:
  HeadBlock(Graph* graph, std::shared_ptr<Tensor> m_output, uint32_t embd, uint32_t vocab,
            UserConfig model_config, Device* device, const std::string& name);

  void execute(WorkSpace* workspace, uint32_t nr_past, bool is_prefill = false) override;

 private:
  uint32_t m_embd;
  uint32_t m_vocab;
  Graph* m_graph;
};

class EmbdBlock : public OprBlockBase {
 public:
  EmbdBlock(Graph* graph, std::shared_ptr<Tensor> m_output, uint32_t embd, uint32_t vocab,
            UserConfig model_config, Device* device, const std::string& name);

 private:
  uint32_t m_embd;
  uint32_t m_vocab;
  Graph* m_graph;
};

class Graph : public std::enable_shared_from_this<Graph> {
 public:
  Graph(UserConfig model_config, Device* device, const std::string& name)
      : m_name(name), m_model_config(model_config), m_device(device) {
    m_workspace = std::make_unique<WorkSpace>();
  }

  ~Graph();

  void load(std::shared_ptr<InputFile> fin, const LlmParams& param, ModelType model_type);

  void optimize();

  void execute(std::vector<int32_t> in_token, std::vector<float>& logist, uint32_t nr_past,
               bool prefill = false);

  Device* device() { return m_device; }

  size_t get_workspace_in_byte();

  template <typename OpBlock, typename... Args>
  std::shared_ptr<Tensor> add_block(Args&&... args) {
    auto block = std::make_shared<OpBlock>(std::forward<Args>(args)...);
    m_blocks.push_back(block);
    return block->output();
  }

  uint32_t get_nr_ctx() { return m_param.n_ctx; }

  uint32_t get_nr_vocab() { return m_param.n_vocab; }

  void reset_ctx();

 private:
  void constuct_llm();

  bool same_input_shape(std::vector<int32_t> in_token);

  std::string m_name;
  LlmParams m_param;
  UserConfig m_model_config;
  Device* m_device = nullptr;

  std::shared_ptr<Tensor> m_embeddings;
  std::unique_ptr<WorkSpace> m_workspace;
  std::unordered_map<std::string, std::shared_ptr<Tensor>> m_weights_map;
  std::vector<std::shared_ptr<OprBlockBase>> m_blocks;
  std::shared_ptr<Tensor> m_input;
  std::shared_ptr<Tensor> m_output;
};
}  // namespace llm_learning

#endif  // SRC_CORE_GRAPH_H_
