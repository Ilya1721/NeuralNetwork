#include "MultipleLayerPerceptron.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "ImprovementWatcher.h"

namespace
{
  double sigmoid(double y)
  {
    return 1.0 / (1.0 + std::exp(-y));
  }

  double sigmoidDerivative(double s)
  {
    return s * (1 - s);
  }

  std::vector<double> scoresToProbabilities(const std::vector<double>& scores)
  {
    std::vector<double> probabilities;
    double maxScore = *std::max_element(scores.begin(), scores.end());
    double sum = 0.0;

    for (const auto& score : scores)
    {
      auto probability = std::exp(score - maxScore);
      probabilities.push_back(probability);
      sum += probability;
    }

    for (auto& probability : probabilities)
    {
      probability /= sum;
    }

    return probabilities;
  }
}  // namespace

int classProbabilitiesToIndex(const std::vector<double>& eachClassProbability)
{
  auto maxProbabilityIt =
    std::max_element(eachClassProbability.begin(), eachClassProbability.end());
  auto classIndex = std::distance(eachClassProbability.begin(), maxProbabilityIt);

  return classIndex;
}

MLPHiddenLayer::MLPHiddenLayer(
  size_t inputDimension,
  size_t perceptronsAmount,
  double lineChangeRate,
  const std::function<double(double)>& sideOfLineFunc
)
{
  for (size_t i = 0; i < perceptronsAmount; ++i)
  {
    mPerceptrons.emplace_back(inputDimension, lineChangeRate, sideOfLineFunc);
  }
}

std::vector<double> MLPHiddenLayer::sidesOfLinesForPoint(
  const std::vector<double>& point, bool useSideOfLineFunc
) const
{
  std::vector<double> sidesOfLines;
  for (const auto& perceptron : mPerceptrons)
  {
    sidesOfLines.emplace_back(perceptron.sideOfLineForPoint(point, useSideOfLineFunc));
  }

  return sidesOfLines;
}

std::vector<double> MLPHiddenLayer::eachPerceptronSlopeChange(
  const MLPHiddenLayer& nextLayer,
  const std::vector<double>& nextLayerDeviations,
  const std::vector<double>& currLayerSidesOfLines
) const
{
  std::vector<double> slopeChanges;
  for (size_t perceptronIdx = 0; perceptronIdx < mPerceptrons.size(); ++perceptronIdx)
  {
    auto nextLayerContribution =
      nextLayer.slopesAffectedDeviation(nextLayerDeviations, perceptronIdx);
    auto currLayerDerivative = sigmoidDerivative(currLayerSidesOfLines[perceptronIdx]);
    auto slopeChange = nextLayerContribution * currLayerDerivative;
    slopeChanges.push_back(slopeChange);
  }

  return slopeChanges;
}

double MLPHiddenLayer::slopesAffectedDeviation(
  const std::vector<double>& nextLayerDeviations, size_t currPerceptronIdx
) const
{
  double deviation = 0.0;
  for (size_t perceptronIdx = 0; perceptronIdx < mPerceptrons.size(); ++perceptronIdx)
  {
    const auto& slopes = mPerceptrons[perceptronIdx].getSlopes();
    deviation += nextLayerDeviations[perceptronIdx] * slopes[currPerceptronIdx];
  }

  return deviation;
}

void MLPHiddenLayer::updateSlopeAndBias(
  const std::vector<double>& point, const std::vector<double>& slopeChanges
)
{
  for (size_t percepIdx = 0; percepIdx < mPerceptrons.size(); ++percepIdx)
  {
    mPerceptrons[percepIdx].updateSlopeAndBias(point, slopeChanges[percepIdx]);
  }
}

MultipleLayerPerceptron::MultipleLayerPerceptron(
  size_t inputDimension,
  size_t classesAmount,
  size_t hiddenLayersAmount,
  size_t perceptronsInFirstLayer,
  double lineChangeRate
)
{
  auto prevLayerPerceptrons = inputDimension;
  auto currLayerPerceptrons = perceptronsInFirstLayer;
  for (size_t layerIdx = 0, pow = 1; layerIdx < hiddenLayersAmount; ++layerIdx, ++pow)
  {
    mLayers.emplace_back(
      prevLayerPerceptrons, currLayerPerceptrons, lineChangeRate, sigmoid
    );
    prevLayerPerceptrons = currLayerPerceptrons;
    currLayerPerceptrons = perceptronsInFirstLayer / std::pow(2, pow);
  }
  mLayers.emplace_back(prevLayerPerceptrons, classesAmount, lineChangeRate, sigmoid);
}

int MultipleLayerPerceptron::classOf(const std::vector<double>& point) const
{
  auto layersSidesOfLines = this->layersSidesOfLines(point);
  auto eachClassProbability = this->eachClassProbability(layersSidesOfLines);
  return classProbabilitiesToIndex(eachClassProbability);
}

std::vector<double> MultipleLayerPerceptron::eachClassProbability(
  const std::vector<std::vector<double>>& layersSidesOfLines
) const
{
  const auto& outputLayerSidesOfLines = layersSidesOfLines.back();
  return scoresToProbabilities(outputLayerSidesOfLines);
}

std::vector<double> MultipleLayerPerceptron::outputLayerSlopeChanges(
  const std::vector<double>& eachClassProbability,
  const std::vector<double>& eachClassCorrectProbability
) const
{
  std::vector<double> slopeChanges;
  for (size_t probIdx = 0; probIdx < eachClassProbability.size(); ++probIdx)
  {
    slopeChanges.push_back(
      eachClassCorrectProbability[probIdx] - eachClassProbability[probIdx]
    );
  }

  return slopeChanges;
}

std::vector<std::vector<double>> MultipleLayerPerceptron::layersSidesOfLines(
  const std::vector<double>& point
) const
{
  std::vector<std::vector<double>> layersSidesOfLines = {point};
  for (size_t layerIdx = 0; layerIdx < mLayers.size() - 1; ++layerIdx)
  {
    layersSidesOfLines.emplace_back(
      mLayers[layerIdx].sidesOfLinesForPoint(layersSidesOfLines.back(), true)
    );
  }
  layersSidesOfLines.emplace_back(
    mLayers.back().sidesOfLinesForPoint(layersSidesOfLines.back(), false)
  );

  return layersSidesOfLines;
}

void MultipleLayerPerceptron::train(const std::vector<Sample>& originalSamples)
{
  std::mt19937 randomNumberGenerator(std::random_device {}());
  std::vector<Sample> samples = originalSamples;
  MLPImprovementWatcher watcher(5);
  while (!watcher.improvementStopped())
  {
    std::shuffle(samples.begin(), samples.end(), randomNumberGenerator);
    std::vector<std::vector<double>> pointsEachClassProbability;
    std::vector<std::vector<double>> pointsEachClassCorrectProbability;
    for (const auto& [point, eachClassCorrectProbability] : samples)
    {
      auto layersSidesOfLines = this->layersSidesOfLines(point);
      const auto& eachClassProbability = this->eachClassProbability(layersSidesOfLines);
      std::vector<std::vector<double>> layersPerceptronsSlopeChange(mLayers.size());
      layersPerceptronsSlopeChange.back() =
        outputLayerSlopeChanges(eachClassProbability, eachClassCorrectProbability);
      pointsEachClassProbability.push_back(eachClassProbability);
      pointsEachClassCorrectProbability.push_back(eachClassCorrectProbability);

      for (long long layerIdx = mLayers.size() - 2; layerIdx >= 0; --layerIdx)
      {
        auto nextLayerIdx = layerIdx + 1;
        const auto& currentLayer = mLayers[layerIdx];
        const auto& nextLayer = mLayers[nextLayerIdx];
        layersPerceptronsSlopeChange[layerIdx] = currentLayer.eachPerceptronSlopeChange(
          nextLayer, layersPerceptronsSlopeChange[nextLayerIdx],
          layersSidesOfLines[layerIdx + 1]
        );
      }

      for (size_t layerIdx = 0; layerIdx < mLayers.size(); ++layerIdx)
      {
        mLayers[layerIdx].updateSlopeAndBias(
          layersSidesOfLines[layerIdx], layersPerceptronsSlopeChange[layerIdx]
        );
      }
    }
    watcher.update(pointsEachClassProbability, pointsEachClassCorrectProbability);
  }
}
