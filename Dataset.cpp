#include "Dataset.h"

#include <filesystem>
#include <functional>
#include <iostream>

#include "ImageLoader.h"

namespace fs = std::filesystem;

namespace
{
  using SampleFunc =
    void(const std::vector<double>& input, const std::vector<double> target);

  std::vector<double> targetForDigit(int digit)
  {
    std::vector<double> target(10, 0.0);
    target[digit] = 1.0;
    return target;
  }

  void walkImages(
    const std::string& root, int maxImagesPerClass, const std::function<SampleFunc>& func
  )
  {
    for (int digit = 0; digit <= 9; ++digit)
    {
      auto digitImagesFolder = root + "\\" + std::to_string(digit);
      int imagesPerClass = 0;
      for (const auto& digitImage : fs::directory_iterator(digitImagesFolder))
      {
        if (imagesPerClass >= maxImagesPerClass)
        {
          break;
        }
        auto input = loadImage(digitImage.path().string());
        auto target = targetForDigit(digit);
        func(input, target);
        ++imagesPerClass;
      }
    }
  }
}  // namespace

std::vector<Sample> loadDigitsDataset(const std::string& root, int maxImagesPerClass)
{
  std::vector<Sample> dataset;

  walkImages(
    root, maxImagesPerClass,
    [&dataset](const std::vector<double>& input, const std::vector<double> target)
    { dataset.emplace_back(input, target); }
  );

  return dataset;
}

void test(
  const MultipleLayerPerceptron& mlp, const std::string& root, int maxImagesPerClass
)
{
  walkImages(
    root, maxImagesPerClass,
    [&mlp](const std::vector<double>& input, const std::vector<double> target)
    { 
      auto inputClass = mlp.classOf(input);
      auto targetClass = classProbabilitiesToIndex(target);
      std::cout << "Input Digit = " << targetClass;
      std::cout << "; Output Digit = " << inputClass;
      if (inputClass != targetClass)
      {
        std::cout << "; WRONG";
      }
      std::cout << std::endl;
    }
  );
}
