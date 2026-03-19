#pragma once

#include "Perceptron.h"

class SLPHiddenLayer
{
 public:
  SLPHiddenLayer(
    size_t inputDimension,
    size_t outputDimension,
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
    double lineChangeRate
  );

  double sideOfLineForPoint(const std::vector<double>& point) const;
  void train(
    const std::vector<std::vector<double>>& points,
    const std::vector<double>& correctSidesOfLine
  );

 private:
  SLPHiddenLayer mLayer;
  Perceptron mOutput;
};
