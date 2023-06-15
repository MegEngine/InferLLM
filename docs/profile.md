# 模型测速

## 测试方法

考虑到 alpaca 和 LLaMa 的推理过程相同，且中文版仅仅是权重做了调整。我们仅测试 alpaca 中文版和 ChatGLM，结果适用英文模型。

## alpaca 结果

1. 硬件 11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz

    | 模型 | 生成速度(token/s) | 线程数 |
    | :-: | :-: | :-: |
    | chinese-alpaca-7b-q4 | 3.2 | 1 |
    | chinese-alpaca-7b-q4 | 9.2 | 4 |
    | chinese-alpaca-7b-q4 | 10 | 8 |
    | chinese-alpaca-7b-q4 | 9.8 | 16 |

2. 硬件 [AMD EPYC 7742 64-Core @ 2.25GHz](https://www.amd.com/zh-hant/products/cpu/amd-epyc-7742)

    | 模型 | 生成速度(token/s) | 线程数 |
    | :-: | :-: | :-: |
    | chinese-alpaca-7b-q4 | 2.3 | 1 |
    | chinese-alpaca-7b-q4 | 7.3 | 4 |
    | chinese-alpaca-7b-q4 | 10.5 | 8 |
    | chinese-alpaca-7b-q4 | 10.7 | 16 |
    | chinese-alpaca-7b-q4 | 11.2 | 32 |
    | chinese-alpaca-7b-q4 | 12.7 | 64 |

## ChatGLM 结果

1. 硬件 11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz

    | 模型 | 生成速度(token/s) | 线程数 |
    | :-: | :-: | :-: |
    | chatglm-q4 | 3.2 | 1 |
    | chatglm-q4 | 8.0 | 4 |
    | chatglm-q4 | 8.9 | 8 |
    | chatglm-q4 | 7.3 | 16 |

2. 硬件 AMD EPYC 7742 64-Core @ 2.25GHz

    | 模型 | 生成速度(token/s) | 线程数 |
    | :-: | :-: | :-: |
    | chatglm-q4 | 2.4 | 1 |
    | chatglm-q4 | 5.8 | 4 |
    | chatglm-q4 | 8.9 | 8 |
    | chatglm-q4 | 9.1 | 16 |
    | chatglm-q4 | 11.6 | 32 |
    | chatglm-q4 | 11.7 | 64 |
