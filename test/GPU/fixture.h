#pragma once
#include <gtest/gtest.h>

#include "core/op.h"

#include <memory>
#if ENABLE_GPU

namespace inferllm {
namespace test {

class GPU : public ::testing::Test {
public:
    void SetUp() override {
        m_naive = make_unique<CPUDevice>(KernelType::Naive, 1);
        m_device = make_unique<GPUDevice>(0);
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

#endif
