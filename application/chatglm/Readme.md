## 运行 ChatGLM 模型
### 下载 ChatGLM 模型
从 [huggingface](https://huggingface.co/THUDM/chatglm-6b) 上下载模型，并把模型保存为 fp32 的数据格式，直接运行 convert.py 脚本
```
python3 convert.py -o chatglm-fp32.bin
```
> 注意上面脚本运行将消耗大量内存，需要在内存大于 25G 的服务器上运行。如果没有条件，可以直接从 [huggingface](https://huggingface.co/kewin4933/InferLLM-Model/tree/main) 上下载已经量化好的 Int4 模型。
### 量化为 INT4 模型
量化工具是 cpp 编写的，主要源文件是 quantizer.cpp 文件，运行这个文件之前需要编译 InferLLM。
```shell
mkdir build
cd build
cmake ..
make -j4
```
编译完成之后在 build 目录下面有一个 chatglm_quantizer 的可执行文件，通过这个工具可以就上一步中的 chatglm-fp32.bin 模型量化为 INT4 的模型。
```
./chatglm_quantizer path/to/chatglm-fp32.bin chatglm-q4.bin
```
这样就完成了 ChatGLM int4 模型的量化，下面就可以直接运行。运行模型的工具是 build 目录下面的 chatglm 可执行文件。

```
./chatglm -m ./chatglm-q4.bin -t 8
```

至此就完成了 ChatGLM 模型的运行，效果挺好的。
![模型运行](../../asserts/ChatGLM-x86.gif )

