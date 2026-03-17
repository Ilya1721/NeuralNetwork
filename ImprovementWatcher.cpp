#include "ImprovementWatcher.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
  const double MIN_IMPROVEMENT = 1e-5;

  double deviation(double sideOfLine, double correctAnswer)
  {
    const double eps = 1e-12;
    sideOfLine = std::max(eps, std::min(1.0 - eps, sideOfLine));
    return -(
      correctAnswer * std::log(sideOfLine) +
      (1 - correctAnswer) * std::log(1 - sideOfLine)
    );
  }

  double averageDeviation(
    const std::vector<double>& predictions, const std::vector<double>& correctAnswers
  )
  {
    double totalDeviation = 0.0;
    for (size_t predictionIdx = 0; predictionIdx < predictions.size(); ++predictionIdx)
    {
      totalDeviation +=
        deviation(predictions[predictionIdx], correctAnswers[predictionIdx]);
    }

    return totalDeviation / predictions.size();
  }
}  // namespace

ImprovementWatcher::ImprovementWatcher(int passesWithoutImprovementToStop)
  : mPassesWithoutImprovementToStop(passesWithoutImprovementToStop),
    mPassesWithoutImprovement(0),
    mSmallestAverageDeviation(std::numeric_limits<double>::max())
{
}

bool ImprovementWatcher::improvementStopped() const
{
  return mPassesWithoutImprovement >= mPassesWithoutImprovementToStop;
}

void ImprovementWatcher::update(
  const std::vector<double>& predictions, const std::vector<double>& correctAnswers
)
{
  auto currAverageDeviation = averageDeviation(predictions, correctAnswers);
  if (currAverageDeviation < mSmallestAverageDeviation - MIN_IMPROVEMENT)
  {
    mSmallestAverageDeviation = currAverageDeviation;
    mPassesWithoutImprovement = 0;
  }
  else
  {
    ++mPassesWithoutImprovement;
  }
}

MLPImprovementWatcher::MLPImprovementWatcher(int passesWithoutImprovementToPass)
  : ImprovementWatcher(passesWithoutImprovementToPass)
{
}

void MLPImprovementWatcher::update(
  const std::vector<std::vector<double>>& pointsEachClassProbability,
  const std::vector<std::vector<double>>& pointsEachClassCorrectProbability
)
{
  std::vector<double> predictions;
  std::vector<double> correctAnswers;
  convertMLPToSLP(
    pointsEachClassProbability, pointsEachClassCorrectProbability, predictions,
    correctAnswers
  );
  ImprovementWatcher::update(predictions, correctAnswers);
}

void MLPImprovementWatcher::convertMLPToSLP(
  const std::vector<std::vector<double>>& pointsEachClassProbability,
  const std::vector<std::vector<double>>& pointsEachClassCorrectProbability,
  std::vector<double>& predictions,
  std::vector<double>& correctAnswers
)
{
  for (size_t pointIdx = 0; pointIdx < pointsEachClassProbability.size(); ++pointIdx)
  {
    const auto& eachClassCorrectProbability = pointsEachClassCorrectProbability[pointIdx];
    const auto& maxIt = std::max_element(
      eachClassCorrectProbability.begin(), eachClassCorrectProbability.end()
    );
    const auto correctClassIdx =
      std::distance(eachClassCorrectProbability.begin(), maxIt);
    correctAnswers.push_back(*maxIt);
    predictions.push_back(pointsEachClassProbability[pointIdx][correctClassIdx]);
  }
}
