#pragma once

#include <vector>

class ImprovementWatcher
{
 public:
  bool improvementStopped() const;

 protected:
  ImprovementWatcher(int passesWithoutImprovementToStop);

 protected:
  double mSmallestAverageLoss;
  int mPassesWithoutImprovement;
  const int mPassesWithoutImprovementToStop;
};

class SLPImprovementWatcher : public ImprovementWatcher
{
 public:
  SLPImprovementWatcher(int passesWithoutImprovementToStop);

  void update(
    const std::vector<double>& predictions, const std::vector<double>& correctAnswers
  );
};

class MLPImprovementWatcher : public ImprovementWatcher
{
 public:
  MLPImprovementWatcher(int passesWithoutImprovementToStop);

  void update(
    const std::vector<std::vector<double>>& eachPointPredictions,
    const std::vector<std::vector<double>>& eachPointTargets
  );
};
