#include "SingleLayerPerceptron.h"

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
}  // namespace

SLPHiddenLayer::SLPHiddenLayer(
  size_t inputDimension,
  size_t outputDimension,
  double lineChangeRate,
  const std::function<double(double)>& sideOfLineFunc
)
{
  auto perceptronsCount = (2.0 / 3.0) * inputDimension + outputDimension;
  for (size_t i = 0; i < perceptronsCount; ++i)
  {
    mPerceptrons.emplace_back(inputDimension, lineChangeRate, sideOfLineFunc);
  }
}

std::vector<double> SLPHiddenLayer::sidesOfLinesForPoint(const std::vector<double>& point
) const
{
  std::vector<double> sidesOfLines;
  for (const auto& perceptron : mPerceptrons)
  {
    sidesOfLines.emplace_back(perceptron.sideOfLineForPoint(point));
  }

  return sidesOfLines;
}

void SLPHiddenLayer::backpropagate(
  const std::vector<double>& sidesOfLine,
  const std::vector<double>& outputSlopes,
  const std::vector<double>& point,
  double outputSlopeChange
)
{
  for (size_t perceptronIdx = 0; perceptronIdx < mPerceptrons.size(); ++perceptronIdx)
  {
    auto sideOfLine = sidesOfLine[perceptronIdx];
    auto slopeChange =
      outputSlopeChange * outputSlopes[perceptronIdx] * sigmoidDerivative(sideOfLine);
    mPerceptrons[perceptronIdx].updateSlopeAndBias(point, slopeChange);
  }
}

SingleLayerPerceptron::SingleLayerPerceptron(
  size_t inputDimension, size_t outputDimension, double lineChangeRate
)
  : mLayer(inputDimension, outputDimension, lineChangeRate, sigmoid),
    mOutput(inputDimension, lineChangeRate, sigmoid)
{
}

double SingleLayerPerceptron::sideOfLineForPoint(const std::vector<double>& point) const
{
  auto sidesOfLines = mLayer.sidesOfLinesForPoint(point);
  return mOutput.sideOfLineForPoint(sidesOfLines);
}

void SingleLayerPerceptron::train(
  const std::vector<std::vector<double>>& points,
  const std::vector<double>& correctSidesOfLine
)
{
  ImprovementWatcher watcher(10);
  while (!watcher.improvementStopped())
  {
    std::vector<double> finalSidesOfLine;
    for (size_t pointIdx = 0; pointIdx < points.size(); ++pointIdx)
    {
      const auto& point = points[pointIdx];
      auto sidesOfLine = mLayer.sidesOfLinesForPoint(point);
      double finalSideOfLine = mOutput.sideOfLineForPoint(sidesOfLine);
      auto sideOfLineDeviation = correctSidesOfLine[pointIdx] - finalSideOfLine;
      auto slopeChange = sideOfLineDeviation * sigmoidDerivative(finalSideOfLine);
      mLayer.backpropagate(sidesOfLine, mOutput.getSlopes(), point, slopeChange);
      mOutput.updateSlopeAndBias(point, slopeChange);
      finalSidesOfLine.push_back(finalSideOfLine);
    }
    watcher.update(finalSidesOfLine, correctSidesOfLine);
  }
}
