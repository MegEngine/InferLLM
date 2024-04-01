#include <benchmark/benchmark.h>

#include "bench.h"
#include "core/op.h"
#include "core/tensor.h"
#include "utils.h"

using namespace std;

static void benchmark_elewise(benchmark::State& state) {
  using namespace llm_learning;
  using namespace llm_learning::benchmark;
  auto m_device = std::make_unique<Device>(KernelType::X86, state.range(0));
  NormalRNG rng0(1.f);

  Benchmark<Embedding> benchmark(m_device.get());
  benchmark.set_dtype(0, DType::Int32);
  benchmark.set_rng(0, &rng0).create_opr(128u, 10u, DType::Float32);
  benchmark.set_epsilon(1e-3).set_past(0);
  for (auto _ : state) {
    benchmark.exec({TensorShape{128, 128}});
  }
}

BENCHMARK(benchmark_elewise)->RangeMultiplier(2)->Range(2, 28);
