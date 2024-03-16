#include "network.h"

#include <torch/torch.h>
#include <iostream>

using namespace std;
using namespace torch;

int main() {
  Net network(50, 10);
  Tensor x, out;
  x = torch::randn({2, 50});
  out = network->forward(x);
  cout << out << endl;
}
