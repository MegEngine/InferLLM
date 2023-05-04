从 LLM 火爆以来，社区已经出现了非常多优秀的模型，当然他们最大的特点就是体积大，最近为了让大模型可以在更低端的设备上运行，社区做了非常多的工作，  https://github.com/IST-DASLab/gptq 实现了将模型进行低比特量化，因此降低了运行大模型对CPU内存，GPU显存的要求，https://github.com/ggerganov/llama.cpp 实现了在本地 CPU/GPU 上就可以运行大模型，并且步骤非常简单，[replit/replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b) 用更小的模型实现了更智能的 code 生成。可以看到模型的小型化和轻量部署也是一个大模型的发展方向。

鉴于此，MegEngine 团队开发了 InferLLM 工程，主要目的是提供大模型推理的学习框架，大家可以直观的了解大模型推理的具体细节。相比 llama.cpp 工程，InferLLM 结构更简单，避免将所有逻辑代码和 kernel 代码放在一个文件中，同时还存在非常多的宏，llama.cpp 对于学习和二次开发不是很友好，InferLLM 也是主要借鉴 llama.cpp，如：使用 llama.cpp 的模型格式，以及 copy 了一些 code，同时 InferLLM 对其进行了重构，使得代码更简单直接，非常容易上手，框架代码和 kernel 代码分开，其实在大模型推理中，真正需要优化的 kernel 是远远小于 CNN 的 kernel 的。

另外 InferLLM 也可以用在生产中，因为它可以在一个性能一般的手机上流畅的运行大模型，可以进行流畅人机对话，目前在手机上运行一个 llama 7b 4bit 的模型，只需要 4G 左右内存，这个内存是现在大多数手机都能满足的。相信在不久之后会出现很多大模型中的轻量化模型，可以直接在端上进行部署和推理，因为目前手机是大家最容易获得的计算资源，没有理由浪费如此庞大的计算集群。

下面是在 xiomi9，Qualcomm SM8150 Snapdragon 855 上使用 4 现成运行 llama 7b 4bit 量化模型的情况：
![](./../asserts/android.gif)

InferLLM 主要由几部分组成

- Model：主要负责输入的 tokenizer，词汇表管理，存储一些历史的 token 以及 Decoder 之后的采样等。
- Graph/Op：负责创建整个模型，包括模型的中 Op 直接的连接关系，Op 的执行，以及 Op 输入输出等内存资源的管理
- Kernel：提供不同后端优化的 Kernel，目前包括 x86，Arm，naive，当 x86 和 Arm 中没有优化的 Kernel，会直接 fallback 到 naive 中进行运行

InferLLM 的支持以下功能：

- 支持每个 Op 执行前准备资源，每个 Op 执行前都需要调用 pre_execute，执行之后调用 end_execute。这样可以方便在内存不足的设备上，在执行前从磁盘中间权重读取到 RAM 中，执行完成之后将权重存回磁盘中，也可以直接使用 mmap，让操作系统自动处理这些逻辑
- 支持每一个 Multi-Head Attention 的 KV cache，每次计算出来的 Key 和 Value 都保存在 KVStorage 中，KVStorage 支持通过 token 的 id 索引，另外如果 KV 的 cache 过大时，还支持将其 swap 出去
- 支持 CPU 上多线程，SIMD，量化等加速，多线程是自己实现的一个类似 OpenMP 静态调度的逻辑，使用无锁的线程池来进行多线程之间的同步
- 可以支持多种模型格式，目前仅仅支持了 llama 类似的模型，未来将支持更多的模型结构