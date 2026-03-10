#pragma once

#include "Perceptron.h"

class HiddenLayer
{
 public:
  HiddenLayer(
    size_t inputDimension,
    size_t outputDimension,
    double yIntercept,
    double lineChangeRate,
    const std::function<double(double)>& sideOfLineFunc
  );

  std::vector<double> sidesOfLinesForPoint(const std::vector<double>& point) const;
  void backpropagate(
    const std::vector<double>& sidesOfLine,
    const std::vector<double>& outputSlopes,
    const std::vector<double>& point,
    double outputSlopeChange
  );

 private:
  std::vector<Perceptron> mPerceptrons;
};

class SingleLayerPerceptron
{
 public:
  SingleLayerPerceptron(
    size_t inputDimension,
    size_t outputDimension,
    double yIntercept,
    double lineChangeRate
  );

  double sideOfLineForPoint(const std::vector<double>& point) const;
  void train(
    const std::vector<std::vector<double>>& points,
    const std::vector<double>& correctSidesOfLine,
    int passesCount
  );

 private:
  HiddenLayer mLayer;
  Perceptron mOutput;
};
