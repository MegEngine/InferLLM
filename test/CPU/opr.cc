
#include "check.h"
#include "fixture.h"

using namespace std;
using namespace llm_learning;
using namespace test;

TEST_F(CPU, TestEmbeding) {
  Checker<Embedding> checker(device(), naive_device());
  checker.set_dtype(0, DType::Int32).set_dtype(1, DType::Int4);
  NormalRNG rng0(1.f);
  UIntRNG rng1(1, 9);
  checker.set_rng(0, &rng0);
  checker.set_rng(1, &rng1);
  checker.create_opr(128u, 10u, DType::Float32);
  checker.set_epsilon(1e-3)
      .set_past(0)  //
      .exec({TensorShape{4}});
}