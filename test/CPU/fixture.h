#pragma once
#include <gtest/gtest.h>

#include "core/op.h"

#include <memory>

namespace inferllm {
namespace test {

class CPU : public ::testing::Test {
public:
    void SetUp() override {
#if INFER_X86
        m_device = make_unique<CPUDevice>(KernelType::X86, 2);
#elif INFER_ARM
        m_device = make_unique<CPUDevice>(KernelType::Arm, 2);
#endif
        m_naive = make_unique<CPUDevice>(KernelType::Naive, 1);
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
}  // namespace inferllm
