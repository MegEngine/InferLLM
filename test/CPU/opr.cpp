
#include "checker.h"
#include "fixture.h"

using namespace std;
using namespace inferllm;
using namespace test;

TEST_F(CPU, TestEmbeding) {
    Checker<Embedding> checker(device(), naive_device());
    checker.set_dtype(0, DType::Int32).set_dtype(1, DType::Int4);
    NormalRNG rng0(1.f);
    UIntRNG rng1(1, 9);
    checker.set_rng(0, &rng0);
    checker.set_rng(1, &rng1);
    checker.create_opr(128u, 10u, DType::Float32);
    checker.set_epsilon(1e-3).set_past(0).exec({TensorShape{4}});
}

TEST_F(CPU, TestMatMul) {
    Checker<MatMul> checker(device(), naive_device());
    size_t K = 4096;
    for (DType dtype : {DType::Int4}) {
        for (bool packed : {true, false}) {
            checker.set_weight_dtype(0, dtype);
            checker.weight_packed(packed);
            checker.profile(20);
            for (bool bias : {false, true}) {
                for (size_t N : {16384, 4096}) {
                    checker.create_opr(N, K, bias);
                    for (size_t M : {1}) {
                        auto time = checker.exec({TensorShape{M, K}});
                        printf("test case M=%zu, N=%zu, K=%zu, bias=%d, packed=%d, "
                               "time=%f\n",
                               M, N, K, bias, packed, time);
                    }
                }
            }
        }
    }
}

TEST_F(CPU, TestLlamaAttention) {
    Checker<LlamaAttention> checker(device(), naive_device());
    checker.weight_packed(true);
    for (auto dtype : {DType::Int4}) {
        uint32_t ctx = 128;
        uint32_t layer_id = 0;
        checker.set_weight_dtype(0, dtype);
        checker.set_weight_dtype(1, dtype);
        checker.set_weight_dtype(2, dtype);
        for (RotMode mode : {RotMode::Mode0, RotMode::ModelRotHalf}) {
            for (uint32_t seqlen : {2, 5}) {
                for (uint32_t dim : {128, 256}) {
                    for (uint32_t head : {16, 32}) {
                        for (uint32_t rot : {dim / head}) {
                            checker.create_opr(
                                    dim, rot, ctx, head, layer_id, DType::Float32,
                                    mode);
                            checker.exec({TensorShape{seqlen, dim}});
                        }
                    }
                }
            }
        }
    }
}

TEST_F(CPU, TestGlmAttention) {
    Checker<GlmAttention> checker(device(), naive_device());
    checker.weight_packed(true);
    for (auto dtype : {DType::Float32}) {
        uint32_t ctx = 128;
        uint32_t layer_id = 0;
        checker.set_weight_dtype(0, dtype);
        checker.set_weight_dtype(1, dtype);
        checker.set_weight_dtype(2, dtype);
        for (uint32_t seqlen : {3, 5, 7}) {
            for (uint32_t dim : {128, 256}) {
                for (uint32_t head : {16}) {
                    for (uint32_t rot : {dim / head}) {
                        checker.create_opr(
                                dim, rot, ctx, head, layer_id, DType::Float32,
                                RotMode::Mode0);
                        checker.exec({TensorShape{seqlen, dim}});
                    }
                }
            }
        }
    }
}

TEST_F(CPU, TestGlm2MultiQueryAttention) {
    Checker<Glm2MultiQueryAttention> checker(device(), naive_device());
    checker.weight_packed(true);
    uint32_t q_group = 2;
    for (auto dtype : {DType::Float32, DType::Int4}) {
        uint32_t ctx = 128;
        uint32_t layer_id = 0;
        checker.set_weight_dtype(0, dtype);
        checker.set_weight_dtype(1, dtype);
        checker.set_weight_dtype(2, dtype);
        for (uint32_t seqlen : {1, 4}) {
            for (uint32_t dim : {128}) {
                for (uint32_t head : {32}) {
                    checker.create_opr(
                            dim, q_group, ctx, head, layer_id, DType::Float32,
                            RotMode::Mode0);
                    checker.exec({TensorShape{seqlen, dim}});
                }
            }
        }
    }
}
