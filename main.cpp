#include <filesystem>

#include "Dataset.h"
#include "MultipleLayerPerceptron.h"

namespace fs = std::filesystem;

int main()
{
  auto root = fs::current_path() / "ThirdParty" / "mnist-png";
  auto trainFolder = root / "train";
  auto samples = loadDigitsDataset(trainFolder.string(), 500);

  MultipleLayerPerceptron mlp(
    samples[0].input.size(), samples[0].target.size(), 2, 128, 0.01
  );
  mlp.train(samples);

  auto testFolder = root / "test";
  test(mlp, testFolder.string(), 1);

  return 0;
}