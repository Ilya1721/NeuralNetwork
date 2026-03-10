#include <iostream>

#include "Perceptron.h"

int sideOfLine(double y)
{
  return y >= 0 ? 1 : 0;
}

int main()
{
  // Training data
  // [income, debt, creditScore, yearsAtJob]
  std::vector<std::vector<double>> inputs = {
    {0.9, 0.1, 0.9, 0.8},
    {0.8, 0.2, 0.8, 0.7},
    {0.2, 0.8, 0.3, 0.2},
    {0.3, 0.7, 0.4, 0.3},
    {0.7, 0.3, 0.7, 0.6}
  };
  std::vector<double> labels = {1, 1, 0, 0, 1};
  Perceptron p(inputs[0].size(), 0.0, 0.1, sideOfLine);
  p.train(inputs, labels, 10);

  std::vector<double> applicant = {0.45, 0.25, 0.8, 0.6};
  auto side = p.sideOfLineForPoint(applicant);
  if (side == 1)
  {
    std::cout << "Approved";
  }
  else
  {
    std::cout << "Rejected";
  }
}