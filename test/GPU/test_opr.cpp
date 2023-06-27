
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
