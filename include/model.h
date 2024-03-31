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

/*!
 * \file model.h
 * \brief Base class for model struct.
 */
#ifndef INCLUDE_MODEL_H_
#define INCLUDE_MODEL_H_

#include <memory>
#include <string>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

namespace llm_learning {

struct ModelConfig {
  std::string device;
  std::string weight_type;  // dtype include 'float32','float16','int8','int4'
  std::string compt_type;
  uint32_t nr_thread;
  bool enable_mmap;
};

class ModelImp;

class API Model {
 public:
  /*!
   * \brief Construct a model struct by the model_name
   * \param config Model config parameter.
   * \param model_name the model struct name. it must be registered 
            internal before load it.
   */
  Model(const ModelConfig& config, const std::string& model_name);

  // load the model from model_path
  void load(const std::string& model_path);

  // allocate memory for the model or init its param
  void init(uint32_t top_k, float top_p, float temp, float repeat_penalty, int repeat_last_n,
            int32_t seed);

  // get the remain token number
  uint32_t get_remain_token();

  // reset the token
  void reset_token();

  // prefill the model with inference with the given promote
  void prefill(const std::string& promote);

  // decode the answer one by one
  std::string decode(const std::string& user_input, int& token);

  // decode the answer one by one
  std::string decode_iter(int& token);

 private:
  std::shared_ptr<ModelImp> m_model_imp;
};

}  // namespace llm_learning

#endif  // INCLUDE_MODEL_H_
