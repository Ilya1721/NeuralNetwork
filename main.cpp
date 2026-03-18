#include <filesystem>
#include <iostream>

#include "Dataset.h"
#include "MultipleLayerPerceptron.h"

namespace fs = std::filesystem;

int main()
{
  auto root = fs::current_path() / "ThirdParty" / "mnist-png";
  auto trainFolder = root / "train";
  auto samples = loadDigitsDataset(trainFolder.string(), 10);
  std::vector<std::vector<double>> inputs, targets;
  for (const auto& [input, target] : samples)
  {
    inputs.push_back(input);
    targets.push_back(target);
  }

  MultipleLayerPerceptron mlp(inputs[0].size(), targets[0].size(), 3, 32, 0.0, 0.01);
  mlp.train(inputs, targets);

  auto testFolder = root / "test";
  test(mlp, testFolder.string(), 3);

  return 0;
}