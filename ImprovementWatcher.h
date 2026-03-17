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

 protected:
  double mSmallestAverageDeviation;
  int mPassesWithoutImprovement;
  const int mPassesWithoutImprovementToStop;
};

class MLPImprovementWatcher : public ImprovementWatcher
{
 public:
  MLPImprovementWatcher(int passesWithoutImprovementToStop);

  void update(
    const std::vector<std::vector<double>>& pointsEachClassProbability,
    const std::vector<std::vector<double>>& pointsEachClassCorrectProbability
  );

 private:
  void convertMLPToSLP(
    const std::vector<std::vector<double>>& pointsEachClassProbability,
    const std::vector<std::vector<double>>& pointsEachClassCorrectProbability,
    std::vector<double>& predictions,
    std::vector<double>& correctAnswers
  );
};
