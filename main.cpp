#include <iostream>

#include "SingleLayerPerceptron.h"

int main()
{
  // [income, debt, creditScore]
  std::vector<std::vector<double>> applicants = {
    {0.2, 0.8, 0.3},  // low income, high debt -> reject
    {0.9, 0.2, 0.4},  // high income, low debt -> approve
    {0.3, 0.4, 0.9},  // great credit score -> approve
    {0.4, 0.7, 0.2},  // bad profile -> reject
    {0.8, 0.3, 0.6},  // good profile -> approve
    {0.3, 0.8, 0.8}   // high debt even with good score -> reject
  };
  std::vector<double> correctAnswers = {0, 1, 1, 0, 1, 0};

  SingleLayerPerceptron slp(3, 1, 0.0, 0.1);
  slp.train(applicants, correctAnswers, 5000);

  std::vector<double> applicant = { 0.45, 0.4, 0.5 };
  double sideOfLine = slp.sideOfLineForPoint(applicant);

  if (sideOfLine > 0.5)
  {
    std::cout << "Loan APPROVED\n";
  }
  else
  {
    std::cout << "Loan REJECTED\n";
  }

  return 0;
}