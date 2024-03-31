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

#include "graph.h"

#include <sys/time.h>

#include <fstream>
#include <vector>

using namespace llm_learning;

void OprBlockBase::deduce_output_shape() {
  for (auto opr : m_oprs) {
    opr->deduce_output_shape();
  }
}

void OprBlockBase::execute(WorkSpace* workspace, uint32_t nr_past, bool) {
  for (auto opr : m_oprs) {
    opr->pre_execute();
#ifdef INFER_PROFILE
    struct timeval start, end;
    gettimeofday(&start, NULL);
#endif
    opr->execute(workspace, nr_past);
#ifdef INFER_PROFILE
    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    float micros = (seconds * 1000) + (float)(end.tv_usec - start.tv_usec) / 1000;
    printf("Op %s spent time %f ms\n", opr->name().c_str(), micros);
#endif
    opr->end_execute();
  }
}

size_t OprBlockBase::get_workspace_in_byte() {
  size_t max_workspace = 0;
  for (auto opr : m_oprs) {
    size_t workspace = opr->get_workspace_in_byte();
    max_workspace = max_workspace < workspace ? workspace : max_workspace;
  }
  return max_workspace;
}

OprBlockBase::OprBlockBase(std::shared_ptr<Tensor> input, Device* device, const std::string& name)
    : m_name(name), m_device(device), m_input(input) {}

AttentionBlock::AttentionBlock(Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd,
                               uint32_t head, uint32_t rot, uint32_t nr_ctx,
                               UserConfig model_config, Device* device, const std::string& name)
    : OprBlockBase(input, device, name), m_embd(embd), m_head(head), m_rot(rot), m_graph(graph) {
  INFER_ASSERT(embd % head == 0, "Embedding and head is not match.");
  m_index = 0;
  m_kstorage = std::make_unique<KvStorage>(std::vector<size_t>{nr_ctx, embd},
                                           model_config.compt_type, device);
  m_vstorage = std::make_unique<KvStorage>(std::vector<size_t>{nr_ctx, embd},
                                           model_config.compt_type, device);
  //! LayerNorm
  auto inputs = add_opr<LayerNorm>(device, name + "_norm", OpIOs{input}, m_embd, true);
  //! kqv-matmul
  auto v_out = add_opr<Attention>(device, name, inputs, embd, rot, nr_ctx, head,
                                  model_config.weight_type, m_kstorage.get(), m_vstorage.get())[0];
  //! matmul proj
  auto proj_out = add_opr<MatMul>(device, name + ".wo", OpIOs{v_out},
                                  std::vector<size_t>{embd, embd}, model_config.weight_type)[0];
  //! elemwise add
  auto output = add_opr<Elemwise>(device, name + ":Elemwise", OpIOs{this->input(), proj_out},
                                  ElemMode::Add)[0];
  set_output(output);
}

FeedForwardBlock::FeedForwardBlock(Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd,
                                   uint32_t mult, UserConfig model_config, Device* device,
                                   const std::string& name)
    : OprBlockBase(input, device, name), m_embd(embd), m_graph(graph) {
  size_t nff = ((2 * (4 * embd) / 3 + mult - 1) / mult) * mult;
  //! LayerNorm
  auto norm_out = add_opr<LayerNorm>(device, name + ".ffn_norm", OpIOs{input}, m_embd, true)[0];
  //! matmul0
  auto matmul_out0 = add_opr<MatMul>(device, name + ".feed_forward.w3", OpIOs{norm_out},
                                     std::vector<size_t>{nff, embd}, model_config.weight_type)[0];
  //! matmul1
  auto matmul_out1 = add_opr<MatMul>(device, name + ".feed_forward.w1", OpIOs{norm_out},
                                     std::vector<size_t>{nff, embd}, model_config.weight_type)[0];
  //! silu activation
  auto silu_out = add_opr<Elemwise>(device, name + "_silu", OpIOs{matmul_out1}, ElemMode::Silu)[0];
  //! elemwise mul
  auto mul_out =
      add_opr<Elemwise>(device, name + "_elemwise", OpIOs{silu_out, matmul_out0}, ElemMode::Mul)[0];
  //! matmul2
  auto matmul_out2 = add_opr<MatMul>(device, name + ".feed_forward.w2", OpIOs{mul_out},
                                     std::vector<size_t>{embd, nff}, model_config.weight_type)[0];
  //! elemwise add
  auto output =
      add_opr<Elemwise>(device, name + "_add", OpIOs{this->input(), matmul_out2}, ElemMode::Add)[0];
  set_output(output);
}

HeadBlock::HeadBlock(Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t vocab,
                     UserConfig model_config, Device* device, const std::string& name)
    : OprBlockBase(input, device, name), m_embd(embd), m_graph(graph) {
  //! LayerNorm
  auto norm_out = add_opr<LayerNorm>(device, name + "norm", OpIOs{input}, m_embd, true)[0];
  //! matmul
  auto matmul_out =
      add_opr<MatMulLast>(device, name + "output", OpIOs{norm_out},
                          std::vector<size_t>{vocab, embd}, model_config.weight_type)[0];
  set_output(matmul_out);
}

void HeadBlock::execute(WorkSpace* workspace, uint32_t nr_past, bool is_prefill) {
  //! prefill is no need to execute
  if (!is_prefill) {
    for (auto opr : oprs()) {
      opr->pre_execute();
      opr->execute(workspace, nr_past);
      opr->end_execute();
    }
  }
}

EmbdBlock::EmbdBlock(Graph* graph, std::shared_ptr<Tensor> input, uint32_t embd, uint32_t vocab,
                     UserConfig model_config, Device* device, const std::string& name)
    : OprBlockBase(input, device, name), m_embd(embd), m_graph(graph) {
  auto embd_out = add_opr<Embedding>(OpIOs{input}, embd, vocab, model_config.compt_type, device,
                                     "tok_embeddings")[0];
  set_output(embd_out);
}

void Graph::load(std::shared_ptr<InputFile> fin, const LlmParams& param,
                 llm_learning::ModelType model_type) {
  m_param = param;
  m_input = std::make_shared<Tensor>(m_device, m_name + ":input");

  Graph::constuct_llm();

  optimize();

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
    INFER_ASSERT(m_weights_map.count(name) == 1, "Error weight is not found when loading.");
    if (model_type >= ModelType::LLAMA_FILE_VERSION_GGJT_V1) {
      // skip to the next multiple of 32 bytes
      fin->skip(-fin->tell() & 31);
    }
    auto weight = m_weights_map[name];
    INFER_ASSERT(weight->length() == nr_number, "Error length of weight is mismatch.");
    weight->set_file(fin, fin->tell());
    fin->skip(weight->length_in_byte());
  }
}

void Graph::constuct_llm() {
  std::shared_ptr<Tensor> input = m_input;
  //! embd
  input = add_block<EmbdBlock>(this, input, m_param.n_embd, m_param.n_vocab, m_model_config,
                               m_device, "");

  int nr_layer = m_param.n_layer;
  for (int i = 0; i < nr_layer; i++) {
    std::string name = "layers." + std::to_string(i);
    input = add_block<AttentionBlock>(this, input, m_param.n_embd, m_param.n_head, m_param.n_rot,
                                      m_param.n_ctx, m_model_config, m_device, name + ".attention");

    input = add_block<FeedForwardBlock>(this, input, m_param.n_embd, m_param.n_mult, m_model_config,
                                        m_device, name);
  }
  //! the last layer
  m_output = add_block<HeadBlock>(this, input, m_param.n_embd, m_param.n_vocab, m_model_config,
                                  m_device, "");
  //! collect all the weights
  size_t total_data = 0;
  for (auto block : m_blocks) {
    auto all_weights = block->get_all_weights();
    for (auto weight : all_weights) {
      std::string name = weight->name();
      INFER_ASSERT(m_weights_map.count(name) == 0, "dumplicated weight.");
      m_weights_map[name] = weight;
      total_data += weight->length_in_byte();
    }
  }
  INFER_LOG("total weights size is %zu\n", total_data);
}

size_t Graph::get_workspace_in_byte() {
  size_t max_workspace = 0;
  for (size_t i = 0; i < m_blocks.size(); i++) {
    m_blocks[i]->deduce_output_shape();
    size_t workspace = m_blocks[i]->get_workspace_in_byte();
    max_workspace = workspace > max_workspace ? workspace : max_workspace;
  }
  return max_workspace;
}

void Graph::execute(std::vector<int32_t> in_token, std::vector<float>& logist, uint32_t nr_past,
                    bool prefill) {
  if (!same_input_shape(in_token)) {
    m_input->set_shape({in_token.size()}, DType::Int32);
    if (m_workspace->ptr()) {
      m_device->free_device(m_workspace->ptr());
    }
    size_t len = get_workspace_in_byte();
    if (len > 0) {
      auto data = m_device->allocate(len);
      m_workspace->set_memory(data, len);
    }
  }
  m_input->set_shared_memory(in_token.data(), in_token.size() * sizeof(int32_t));

  INFER_ASSERT(m_output->length() == logist.size(), "output length is not match with logist size");
  m_output->set_shared_memory(logist.data(), logist.size() * sizeof(float));

  for (size_t i = 0; i < m_blocks.size(); i++) {
    m_blocks[i]->execute(m_workspace.get(), nr_past, prefill);
  }
}

void Graph::reset_ctx() {
  for (size_t i = 0; i < m_blocks.size(); i++) {
    m_blocks[i]->reset_ctx();
  }
}

void Graph::optimize() {}

Graph::~Graph() {
  if (m_workspace->ptr()) {
    m_device->free_device(m_workspace->ptr());
  }
}

bool Graph::same_input_shape(std::vector<int32_t> in_token) {
  if (m_input->dims() != in_token.size()) {
    return false;
  } else {
    for (size_t i = 0; i < in_token.size(); i++) {
      if (m_input->shape()[i] != in_token[i]) {
        return false;
      }
    }
  }
  return true;
}
