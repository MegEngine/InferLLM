
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
