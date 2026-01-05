#include <cmath>
#include <vector>

double relu(double x) {
  if (x > 0.0) {
    return x;
  } else
    return 0.0;
}

double ddxRelu(double x) {
  if (x > 0.0) {
    return 1.0;
  } else
    return 0.0;
}

std::vector<double> reluVector(const std::vector<double> &input) {
  std::vector<double> output;
  for (double value : input) {
    output.push_back(relu(value));
  }
  return output;
}

std::vector<double> ddxReluVector(const std::vector<double> &input) {
  std::vector<double> output;
  for (double value : input) {
    output.push_back(ddxRelu(value));
  }
  return output;
}
