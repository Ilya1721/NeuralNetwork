#include "MultipleLayerPerceptron.h"

#include <cmath>

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

MLPHiddenLayer::MLPHiddenLayer(
  size_t inputDimension,
  size_t perceptronsAmount,
  double yIntercept,
  double lineChangeRate,
  const std::function<double(double)>& sideOfLineFunc
)
{
  for (size_t i = 0; i < perceptronsAmount; ++i)
  {
    mPerceptrons.emplace_back(inputDimension, yIntercept, lineChangeRate, sideOfLineFunc);
  }
}

std::vector<double> MLPHiddenLayer::sidesOfLinesForPoint(const std::vector<double>& point
) const
{
  std::vector<double> sidesOfLines;
  for (const auto& perceptron : mPerceptrons)
  {
    sidesOfLines.emplace_back(perceptron.sideOfLineForPoint(point));
  }

  return sidesOfLines;
}

std::vector<double> MLPHiddenLayer::eachPerceptronSlopeChange(
  const MLPHiddenLayer& prevLayer,
  const std::vector<double>& prevLayerDeviations,
  const std::vector<double>& prevLayerSidesOfLines,
  size_t currLayerIdx
) const
{
  std::vector<double> slopeChanges;
  for (size_t perceptronIdx = 0; perceptronIdx < mPerceptrons.size(); ++perceptronIdx)
  {
    auto prevLayerDeviation = slopesAffectedDeviation(prevLayerDeviations, currLayerIdx);
    auto prevLayerSideOfLine = prevLayerSidesOfLines[perceptronIdx];
    auto slopeChange = prevLayerDeviation * sigmoidDerivative(prevLayerSideOfLine);
    slopeChanges.push_back(slopeChange);
  }

  return slopeChanges;
}

double MLPHiddenLayer::slopesAffectedDeviation(
  const std::vector<double>& eachPerceptronDeviation, size_t layerIdx
) const
{
  double deviation = 0.0;
  for (size_t perceptronIdx = 0; perceptronIdx < mPerceptrons.size(); ++perceptronIdx)
  {
    const auto& slopes = mPerceptrons[perceptronIdx].getSlopes();
    deviation += eachPerceptronDeviation[perceptronIdx] * slopes[layerIdx];
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
  size_t layersAmount,
  size_t perceptronsInFirstLayer,
  double yIntercept,
  double lineChangeRate
)
{
  for (size_t layerIdx = 0, pow = 0; layerIdx < layersAmount; ++layerIdx, ++pow)
  {
    auto perceptronsAmount = perceptronsInFirstLayer / std::pow(2, pow);
    mLayers.emplace_back(
      inputDimension, perceptronsAmount, yIntercept, lineChangeRate, sigmoid
    );
  }
  mLayers.emplace_back(
    inputDimension, classesAmount, yIntercept, lineChangeRate, sigmoid
  );
}

int MultipleLayerPerceptron::classOf(const std::vector<double>& point) const
{
  auto eachClassProbability = this->eachClassProbability(point);
  auto maxProbabilityIt =
    std::max_element(eachClassProbability.begin(), eachClassProbability.end());
  auto classIndex = std::distance(eachClassProbability.begin(), maxProbabilityIt);

  return classIndex;
}

std::vector<double> MultipleLayerPerceptron::eachClassProbability(
  const std::vector<double>& point
) const
{
  std::vector<double> sidesOfLines = point;
  for (const auto& layer : mLayers)
  {
    sidesOfLines = std::move(layer.sidesOfLinesForPoint(point));
  }

  return scoresToProbabilities(sidesOfLines);
}

std::vector<double> MultipleLayerPerceptron::outputLayerEachClassDeviation(
  const std::vector<double>& eachClassProbability,
  const std::vector<double>& eachClassCorrectProbability
) const
{
  std::vector<double> deviations;
  for (size_t probIdx = 0; probIdx < eachClassProbability.size(); ++probIdx)
  {
    deviations.push_back(
      eachClassCorrectProbability[probIdx] - eachClassProbability[probIdx]
    );
  }

  return deviations;
}

void MultipleLayerPerceptron::train(
  const std::vector<std::vector<double>>& points,
  const std::vector<std::vector<double>>& pointsEachClassCorrectProbability
)
{
  MLPImprovementWatcher watcher(10);
  while (!watcher.improvementStopped())
  {
    for (size_t pointIdx = 0; pointIdx < points.size(); ++pointIdx)
    {
      std::vector<std::vector<double>> layersSidesOfLines = {points[pointIdx]};
      for (const auto& layer : mLayers)
      {
        layersSidesOfLines.emplace_back(
          layer.sidesOfLinesForPoint(layersSidesOfLines.back())
        );
      }

      const auto& outputLayerSidesOfLines = layersSidesOfLines.back();
      const auto& eachClassProbability = scoresToProbabilities(outputLayerSidesOfLines);
      std::vector<std::vector<double>> layersPerceptronsSlopeChange(mLayers.size());
      const auto& eachClassCorrectProbability =
        pointsEachClassCorrectProbability[pointIdx];
      layersPerceptronsSlopeChange.back() =
        outputLayerEachClassDeviation(eachClassProbability, eachClassCorrectProbability);

      for (long long layerIdx = mLayers.size() - 1; layerIdx >= 0; --layerIdx)
      {
        auto prevLayerIdx = layerIdx + 1;
        const auto& currentLayer = mLayers[layerIdx];
        const auto& prevLayer = mLayers[prevLayerIdx];
        layersPerceptronsSlopeChange[layerIdx] = currentLayer.eachPerceptronSlopeChange(
          prevLayer, layersPerceptronsSlopeChange[prevLayerIdx],
          layersSidesOfLines[prevLayerIdx], layerIdx
        );
      }

      for (size_t layerIdx = 0; layerIdx < mLayers.size(); ++layerIdx)
      {
        mLayers[layerIdx].updateSlopeAndBias(
          layersSidesOfLines[layerIdx], layersPerceptronsSlopeChange[layerIdx]
        );
      }
    }
  }
}
