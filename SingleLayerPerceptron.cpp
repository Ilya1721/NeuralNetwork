#include "SingleLayerPerceptron.h"

#include <cmath>

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

HiddenLayer::HiddenLayer(
  size_t pointsCount,
  double yIntercept,
  double lineChangeRate,
  size_t outputCount,
  const std::function<int(double)>& sideOfLineFunc
)
{
  auto perceptronsCount = (2 / 3) * pointsCount + outputCount;
  for (size_t i = 0; i < perceptronsCount; ++i)
  {
    mPerceptrons.emplace_back(pointsCount, yIntercept, lineChangeRate, sideOfLineFunc);
  }
}

std::vector<double> HiddenLayer::sidesOfLinesForPoint(const std::vector<double>& point
) const
{
  std::vector<double> sidesOfLines;
  for (const auto& perceptron : mPerceptrons)
  {
    sidesOfLines.emplace_back(perceptron.sideOfLineForPoint(point));
  }

  return sidesOfLines;
}

void HiddenLayer::backpropagate(
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
  size_t pointsCount, double yIntercept, double lineChangeRate, size_t outputCount
)
  : mLayer(pointsCount, yIntercept, lineChangeRate, outputCount, sigmoid),
    mOutput(pointsCount, yIntercept, lineChangeRate, sigmoid)
{
}

int SingleLayerPerceptron::sideOfLineForPoint(const std::vector<double>& point) const
{
  auto sidesOfLines = mLayer.sidesOfLinesForPoint(point);
  return mOutput.sideOfLineForPoint(sidesOfLines);
}

void SingleLayerPerceptron::train(
  const std::vector<std::vector<double>>& points,
  const std::vector<double>& correctSidesOfLine,
  int passesCount
)
{
  for (int passIdx = 0; passIdx < passesCount; ++passIdx)
  {
    for (size_t pointIdx = 0; pointIdx < points.size(); ++pointIdx)
    {
      const auto& point = points[pointIdx];
      auto sidesOfLine = mLayer.sidesOfLinesForPoint(point);
      double finalSideOfLine = mOutput.sideOfLineForPoint(sidesOfLine);
      auto sideOfLineDeviation = correctSidesOfLine[pointIdx] - finalSideOfLine;
      auto slopeChange = sideOfLineDeviation * sigmoidDerivative(finalSideOfLine);
      mOutput.updateSlopeAndBias(point, slopeChange);
      mLayer.backpropagate(sidesOfLine, mOutput.getSlopes(), point, slopeChange);
    }
  }
}
