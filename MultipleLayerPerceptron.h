#pragma once

#include <vector>

#include "Perceptron.h"

int classProbabilitiesToIndex(const std::vector<double>& eachClassProbability);

class MLPHiddenLayer
{
 public:
  MLPHiddenLayer(
    size_t inputDimension,
    size_t perceptronsAmount,
    double yIntercept,
    double lineChangeRate,
    const std::function<double(double)>& sideOfLineFunc
  );

  std::vector<double> sidesOfLinesForPoint(const std::vector<double>& point) const;
  std::vector<double> eachPerceptronSlopeChange(
    const MLPHiddenLayer& prevLayer,
    const std::vector<double>& prevLayerDeviations,
    const std::vector<double>& prevLayerSidesOfLines,
    size_t currLayerIdx
  ) const;

  void updateSlopeAndBias(
    const std::vector<double>& point, const std::vector<double>& slopeChanges
  );

 private:
  double slopesAffectedDeviation(
    const std::vector<double>& eachPerceptronDeviation, size_t layerIdx
  ) const;

 private:
  std::vector<Perceptron> mPerceptrons;
};

class MultipleLayerPerceptron
{
 public:
  MultipleLayerPerceptron(
    size_t inputDimension,
    size_t classesAmount,
    size_t hiddenLayersAmount,
    size_t perceptronsInFirstLayer,
    double yIntercept,
    double lineChangeRate
  );

  int classOf(const std::vector<double>& point) const;

  void train(
    const std::vector<std::vector<double>>& points,
    const std::vector<std::vector<double>>& pointsEachClassCorrectProbability
  );

 private:
  std::vector<double> eachClassProbability(
    const std::vector<std::vector<double>>& layersSidesOfLines
  ) const;
  std::vector<double> outputLayerEachClassDeviation(
    const std::vector<double>& eachClassProbability,
    const std::vector<double>& eachClassCorrectProbability
  ) const;
  std::vector<std::vector<double>> layersSidesOfLines(const std::vector<double>& point
  ) const;

 private:
  std::vector<MLPHiddenLayer> mLayers;
};
