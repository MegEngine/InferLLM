## 运行 baichuan 模型
### 下载 baichaun 微调之后的模型
[baichaun](https://huggingface.co/baichuan-inc/baichuan-7B) 原始模型没有经过微调不能直接进行对话，因此这里的模型是从 [baichuan-vicuna-7b](https://huggingface.co/fireballoon/baichuan-vicuna-7b) 下载，首先把模型保存为 fp32 的数据格式，这个过程中需要 CPU 的内存比较大，直接运行 convert.py 脚本将得到 fp32 数据类型的模型
```
python3 convert.py -o baichuang-fp32.bin
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
编译完成之后在 build 目录下面有一个 quantizer 的可执行文件，通过这个工具可以就上一步中的 chatglm-fp32.bin 模型量化为 INT4 的模型。
```
./quantizer path/to/baichuan-fp32.bin biachuan-q4.bin
```
这样就完成了 baichaun int4 模型的量化，下面就可以直接使用 InferLLM 运行模型了。运行模型的工具是 build 目录下面的 chat 可执行文件。

```
./chat -m ./baichuan-q4.bin -t 8 --type baichuan
```

### 在手机上运行模型
在手机上运行和在 x86 上运行的模型是同样的，只是需要将 InferLLM 交叉编译为手机上可以运行的可执行文件。这里提供了现成的脚本，可以快速完成编译。
```shell
export NDK_ROOT=/path/to/ndk
./tools/android_build.sh
```
编译好的 InferLLM 相关可执行文件在 build-arm64-v8a 文件下。可以通过 adb 命令将这些文件和模型拷贝到手机上，然后在手机上就可以直接运行了。

至此就完成了 baichuan 模型的运行，效果如下：
![模型运行](../../asserts/baichuan-x86.gif )



