## OpenLLaMA-3B

OpenLLaMA 项目地址：https://github.com/openlm-research/open_llama

### 下载 OpenLLaMA-3B 模型
从 [huggingface](https://huggingface.co/openlm-research/open_llama_3b_600bt_preview/tree/main) 上下载模型，该模型为 fp16 的 pytorch 格式权重

### 量化为 INT4 模型
量化工具是 cpp 编写的，主要源文件是 quantizer.cpp 文件，运行这个文件之前需要编译固定版本的 llama.cpp。
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git reset --hard b608b55
git apply openllama.patch
mkdir build
cd build
cmake ..
make -j
cd ..
python convert.py ${PATH_TO_HUGGINGFACE_OPENLLAMA}/pytorch_model.bin
./build/bin/quantize ${PATH_TO_HUGGINGFACE_OPENLLAMA}/ggml-model-f16.bin ggml-model-q4_0.bin q4_0
```

- 克隆仓库后，需要将 commit 回退到 b608b55，因为 InferLLM 最高只支持 ggjt.v1 格式的模型，而 llama.cpp 目前 (commit: 7552ac586380f202b75b18aa216ecfefbd438d94) 已更新到 ggjt.v3 且不向前兼容
- 回退代码后，需要打上补丁，OpenLLaMa 的 3B 模型的细节配置与 7B 存在不一致，从 pytorch 格式（pytorch_model.bin）转换到 ggjt 格式（ggml-model-f16.bin）时需要特殊处理
- 编译完成之后在 build 目录下面有一个 bin/quantize 的可执行文件，通过这个工具可以将上一步中的 ggml-model-f16.bin 模型量化为 INT4 的模型（ggml-model-q4_0.bin）

### 运行 OpenLLaMA-3B 模型

可以参考本项目 alpaca 的 README, 编译获得 alpaca 可执行文件。
```bash
git clone https://github.com/MegEngine/InferLLM.git
mkdir build
cd build
cmake ..
make -j
```

通过 alpaca 可执行文件可以运行量化好的 OpenLLaMA 模型

```bash
./alpaca -m ggml-model-q4_0.bin -t 4
```

