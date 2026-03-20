#include "ImprovementWatcher.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
  const double MIN_IMPROVEMENT = 1e-2;

  double loss(double prediction, double target)
  {
    const double eps = 1e-12;
    prediction = std::max(eps, std::min(1.0 - eps, prediction));
    return -(target * std::log(prediction) + (1 - target) * std::log(1 - prediction));
  }

  double averageLoss(
    const std::vector<double>& predictions, const std::vector<double>& targets
  )
  {
    double totalLoss = 0.0;
    for (size_t predictionIdx = 0; predictionIdx < predictions.size(); ++predictionIdx)
    {
      totalLoss += loss(predictions[predictionIdx], targets[predictionIdx]);
    }

    return totalLoss / predictions.size();
  }

  double averageLoss(
    const std::vector<std::vector<double>>& eachPointPredictions,
    const std::vector<std::vector<double>>& eachPointTargets
  )
  {
    auto pointsAmount = eachPointPredictions.size();
    double totalLoss = 0.0;

    for (size_t pointIdx = 0; pointIdx < pointsAmount; ++pointIdx)
    {
      const auto& pointPredictions = eachPointPredictions[pointIdx];
      const auto& pointTargets = eachPointTargets[pointIdx];

      auto maxTargetIt = std::max_element(pointTargets.begin(), pointTargets.end());
      auto correctTargetIdx = std::distance(pointTargets.begin(), maxTargetIt);
      auto prediction = pointPredictions[correctTargetIdx];

      totalLoss += -std::log(std::max(prediction, 1e-12));
    }

    return totalLoss / pointsAmount;
  }
}  // namespace

ImprovementWatcher::ImprovementWatcher(int passesWithoutImprovementToStop)
  : mPassesWithoutImprovementToStop(passesWithoutImprovementToStop),
    mPassesWithoutImprovement(0),
    mSmallestAverageLoss(std::numeric_limits<double>::max())
{
}

bool ImprovementWatcher::improvementStopped() const
{
  return mPassesWithoutImprovement >= mPassesWithoutImprovementToStop;
}

SLPImprovementWatcher::SLPImprovementWatcher(int passesWithoutImprovementToStop)
  : ImprovementWatcher(passesWithoutImprovementToStop)
{
}

void SLPImprovementWatcher::update(
  const std::vector<double>& predictions, const std::vector<double>& correctAnswers
)
{
  auto currAverageLoss = averageLoss(predictions, correctAnswers);
  if (currAverageLoss < mSmallestAverageLoss - MIN_IMPROVEMENT)
  {
    mSmallestAverageLoss = currAverageLoss;
    mPassesWithoutImprovement = 0;
  }
  else
  {
    ++mPassesWithoutImprovement;
  }
}

MLPImprovementWatcher::MLPImprovementWatcher(int passesWithoutImprovementToStop)
  : ImprovementWatcher(passesWithoutImprovementToStop)
{
}

void MLPImprovementWatcher::update(
  const std::vector<std::vector<double>>& eachPointPredictions,
  const std::vector<std::vector<double>>& eachPointTargets
)
{
  auto currAverageLoss = averageLoss(eachPointPredictions, eachPointTargets);
  if (currAverageLoss + MIN_IMPROVEMENT < mSmallestAverageLoss)
  {
    mSmallestAverageLoss = currAverageLoss;
    mPassesWithoutImprovement = 0;
  }
  else
  {
    ++mPassesWithoutImprovement;
  }
}
