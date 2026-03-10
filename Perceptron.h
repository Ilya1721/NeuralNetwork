#pragma once

#include <functional>
#include <vector>

class Perceptron
{
 public:
  Perceptron(
    size_t pointsCount,
    double yIntercept,
    double lineChangeRate,
    const std::function<int(double)>& sideOfLineFunc
  );

  double sideOfLineForPoint(const std::vector<double>& point) const;
  const std::vector<double>& getSlopes() const;

  void train(
    const std::vector<std::vector<double>>& points,
    const std::vector<double>& correctSidesOfLine,
    int passesCount
  );
  void updateSlopeAndBias(const std::vector<double>& point, double slopeChange);

 private:
  std::vector<double> mSlopes;
  double mYIntercept;
  double mLineChangeRate;
  std::function<double(double)> mSideOfLineFunc;
};
