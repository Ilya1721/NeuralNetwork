#pragma once

#include <vector>

class ImprovementWatcher
{
 public:
  ImprovementWatcher(int passesWithoutImprovementToStop);
  bool improvementStopped() const;
  void update(
    const std::vector<double>& predictions, const std::vector<double>& correctAnswers
  );

 private:
  double mSmallestAverageDeviation;
  int mPassesWithoutImprovement;
  const int mPassesWithoutImprovementToStop;
};
