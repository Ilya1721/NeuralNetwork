#include "Perceptron.h"

#include <cmath>
#include <random>

namespace
{
  double dotProduct(const std::vector<double>& left, const std::vector<double>& right)
  {
    double result = 0;
    for (int i = 0; i < left.size(); ++i)
    {
      result += left[i] * right[i];
    }

    return result;
  }
}  // namespace

Perceptron::Perceptron(
  size_t pointCoordsCount,
  double lineChangeRate,
  const std::function<double(double)>& sideOfLineFunc
)
  : mYIntercept(), mLineChangeRate(lineChangeRate), mSideOfLineFunc(sideOfLineFunc)
{
  initSlopes(pointCoordsCount);
}

double Perceptron::sideOfLineForPoint(
  const std::vector<double>& point, bool useSideOfLineFunc
) const
{
  auto kx = dotProduct(point, mSlopes);
  auto b = mYIntercept;
  auto y = kx + b;

  return useSideOfLineFunc ? mSideOfLineFunc(y) : y;
}

const std::vector<double>& Perceptron::getSlopes() const
{
  return mSlopes;
}

void Perceptron::train(
  const std::vector<std::vector<double>>& points,
  const std::vector<double>& correctSidesOfLine,
  int passesCount
)
{
  for (int passIdx = 0; passIdx < passesCount; ++passIdx)
  {
    for (int pointIdx = 0; pointIdx < points.size(); ++pointIdx)
    {
      auto sideOfLine = sideOfLineForPoint(points[pointIdx]);
      auto sideOfLineDeviation = correctSidesOfLine[pointIdx] - sideOfLine;
      for (int slopeIdx = 0; slopeIdx < mSlopes.size(); ++slopeIdx)
      {
        mSlopes[slopeIdx] +=
          mLineChangeRate * sideOfLineDeviation * points[pointIdx][slopeIdx];
      }
      mYIntercept += mLineChangeRate * sideOfLineDeviation;
    }
  }
}

void Perceptron::updateSlopeAndBias(const std::vector<double>& point, double slopeChange)
{
  for (size_t slopeIdx = 0; slopeIdx < mSlopes.size(); ++slopeIdx)
  {
    mSlopes[slopeIdx] += mLineChangeRate * slopeChange * point[slopeIdx];
  }
  mYIntercept += mLineChangeRate * slopeChange;
}

void Perceptron::initSlopes(size_t inputDimension)
{
  double limit = std::sqrt(1.0 / inputDimension);
  static std::random_device randomDevice;
  static std::mt19937 randomGenerator(randomDevice());
  std::uniform_real_distribution<double> distribution(-limit, limit);

  mSlopes.resize(inputDimension);
  for (auto& slope : mSlopes)
  {
    slope = distribution(randomGenerator);
  }
}
