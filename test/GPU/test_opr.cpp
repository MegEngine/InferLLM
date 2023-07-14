
#include "checker.h"
#include "fixture.h"

using namespace std;
using namespace inferllm;
using namespace test;

TEST_F(GPU, TestEmbeding) {
    Checker<Embedding> checker(device(), naive_device());
    NormalRNG rng0(1.f);
    UIntRNG rng1(1, 9);
    checker.set_rng(0, &rng1);
    checker.set_weight_rng(0, &rng0);
    checker.create_opr(128u, 10u, DType::Float32);
    for (auto wdtype : {DType::Int4, DType::Float32}) {
        checker.set_dtype(0, DType::Int32).set_weight_dtype(0, wdtype);
        checker.set_epsilon(1e-3).set_past(0).exec({TensorShape{4}});
    }
}

TEST_F(GPU, TestElemwise) {
    Checker<Elemwise> checker(device(), naive_device());
    for (auto mode : {ElemMode::Gelu, ElemMode::Silu}) {
        checker.create_opr(mode, -INFINITY);
        checker.set_dtype(0, DType::Float32);
        checker.exec({TensorShape{4, 10}});
        checker.exec({TensorShape{5, 10, 13}});
    }

    for (auto mode : {ElemMode::Add, ElemMode::Mul}) {
        checker.create_opr(mode, -INFINITY);
        checker.set_dtype(0, DType::Float32).set_dtype(1, DType::Float32);
        checker.exec({TensorShape{4, 10}, TensorShape{4, 10}});
        checker.exec({TensorShape{5, 10, 13}, TensorShape{5, 10, 13}});
    }
}

TEST_F(GPU, TestLayerNorm) {
    Checker<LayerNorm> checker(device(), naive_device());
    checker.set_epsilon(2e-1);
    float eps = 1e-5;
    for (bool mul : {true, false}) {
        for (bool bias : {true, false}) {
            for (bool rms : {true, false}) {
                checker.create_opr(4096, mul, bias, rms, eps);
                for (size_t seqlen : {4}) {
                    checker.exec({TensorShape{seqlen, 4096}});
                }
            }
        }
    }
}

TEST_F(GPU, TestMatMul) {
    Checker<MatMul> checker(device(), naive_device());
    checker.set_epsilon(1e-2);
    size_t K = 128;
    for (DType dtype : {DType::Int4, DType::Float32}) {
        checker.set_weight_dtype(0, dtype);
        if (dtype == DType::Int4) {
            checker.set_epsilon(5e-1);
        }
        for (bool bias : {false, true}) {
            for (size_t N : {2, 128}) {
                checker.create_opr(N, K, bias);
                for (size_t M : {1, 4}) {
                    checker.exec({TensorShape{M, K}});
                }
            }
        }
    }
}

TEST_F(GPU, TestSoftMax) {
    Checker<SoftMax> checker(device(), naive_device());
    checker.create_opr();
    for (size_t M : {1, 2, 32}) {
        for (size_t dim : {1, 128, 4096}) {
            checker.exec({TensorShape{M, dim}});
        }
    }
}

TEST_F(GPU, TestDiagMask) {
    Checker<DiagMask> checker(device(), naive_device());
    checker.create_opr();
    for (size_t head : {32, 36}) {
        for (size_t dim : {13, 128, 256}) {
            checker.exec({TensorShape{head, dim, dim}});
        }
    }
}

TEST_F(GPU, TestLlamaAttention) {
    Checker<LlamaAttention> checker(device(), naive_device());
    for (auto dtype : {DType::Float32/*, DType::Int4*/}) {
        uint32_t ctx = 128;
        uint32_t layer_id = 0;
        checker.set_weight_dtype(0, dtype);
        checker.set_weight_dtype(1, dtype);
        checker.set_weight_dtype(2, dtype);
        for (uint32_t seqlen : {1/*, 4, 32*/}) {
            for (uint32_t dim : {128}) {
                for (uint32_t head : {16}) {
                    for (uint32_t rot : {dim / head}) {
                        checker.create_opr(dim, rot, ctx, head, layer_id, DType::Float32);
                        checker.exec({TensorShape{seqlen, dim}});
                    }
                }
            }
        }
    }
}
