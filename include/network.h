#pragma once

#include <iostream>
#include <torch/torch.h>

class NetImpl : public torch::nn::Module
{
public:
  NetImpl(int fc1_dims, int fc2_dims);

  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::nn::Linear out{nullptr};
};

TORCH_MODULE(Net);