#pragma once

#include <string>
#include <vector>

#include "MultipleLayerPerceptron.h"

std::vector<Sample> loadDigitsDataset(
  const std::string& root, int maxImagesPerClass = 20
);
void test(
  const MultipleLayerPerceptron& mlp, const std::string& root, int maxImagesPerClass = 20
);
