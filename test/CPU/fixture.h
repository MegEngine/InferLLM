#ifndef TEST_CPU_FIXTURE_H_
#define TEST_CPU_FIXTURE_H_
#include <gtest/gtest.h>

#include <memory>

#include "core/device.h"
#include "core/op.h"

namespace llm_learning {
namespace test {

class CPU : public ::testing::Test {
 public:
  void SetUp() override {
#if INFER_X86
    m_device = std::make_unique<Device>(KernelType::X86, 2);
#elif INFER_ARM
    m_device = make_unique<CPUDevice>(KernelType::Arm, 1);
#endif
    m_naive = std::make_unique<Device>(KernelType::Naive, 1);
  }
  void TearDown() override {
    m_device.reset();
    m_naive.reset();
  }

  Device* device() { return m_device.get(); }
  Device* naive_device() { return m_naive.get(); }

 protected:
  std::unique_ptr<Device> m_device;
  std::unique_ptr<Device> m_naive;
};

}  // namespace test
}  // namespace llm_learning

#endif
