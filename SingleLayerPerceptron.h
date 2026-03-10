#pragma once

#include "Perceptron.h"

class HiddenLayer
{
 public:
  HiddenLayer(
    size_t pointsCount,
    double yIntercept,
    double lineChangeRate,
    size_t outputCount,
    const std::function<int(double)>& sideOfLineFunc
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
    size_t pointsCount, double yIntercept, double lineChangeRate, size_t outputCount
  );

  int sideOfLineForPoint(const std::vector<double>& point) const;
  void train(
    const std::vector<std::vector<double>>& points,
    const std::vector<double>& correctSidesOfLine,
    int passesCount
  );

 private:
  HiddenLayer mLayer;
  Perceptron mOutput;
};
