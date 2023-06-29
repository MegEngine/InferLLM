## 运行 ChatGLM/ChatGLM2 模型
### 下载 ChatGLM/ChatGLM2 模型
从 [huggingface chatglm-6b](https://huggingface.co/THUDM/chatglm-6b) 或者 [huggingface chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b) 上下载模型，并把模型保存为 fp32 的数据格式，直接运行 convert.py 脚本，该脚本支持一个 version 参数，可以指定模型的版本，如果不指定，默认下载和转换 chatglm-6b 模型。
* version = 1: chatglm-6b
* version = 2: chatglm2-6b

下面的命令为转换 chatglm2-6b 模型
```
python3 convert.py -o chatglm-fp32.bin -v 2 
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
编译完成之后在 build 目录下面有一个 quantizer 的可执行文件，通过这个工具可以将上一步中的 chatglm-f32.bin/chatglm2-fp32.bin 模型量化为 INT4 的模型。
```
./quantizer path/to/chatglm-fp32.bin chatglm-q4.bin
```
这样就完成了 ChatGLM int4 模型的量化，下面就可以直接运行。运行模型的工具是 build 目录下面的 chatglm 可执行文件。运行模型时候也需要传递一个 version 参数，默认 version 参数为 1，如果需要运行 chatglm2 模型，则传递给 version 参数为 2

下面是运行 chatglm2 模型的命令
```
./chatglm -m ./chatglm2-q4.bin -t 8 -v 2
```

至此就完成了 ChatGLM 模型的运行，效果挺好的。
![模型运行](../../asserts/ChatGLM-x86.gif )

