#include "network.h"

NetImpl::NetImpl(int fc1_dims, int fc2_dims)
: fc1(fc1_dims, fc1_dims)
, fc2(fc1_dims, fc2_dims)
, out(fc2_dims, 1)
{
  register_module("fc1", fc1);
  register_module("fc2", fc2);
  register_module("out", out);
}

torch::Tensor NetImpl::forward(torch::Tensor x)
{
  x = torch::relu(fc1(x));
  x = torch::relu(fc2(x));
  x = out(x);
  return x;
}
