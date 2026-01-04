#include <algorithm>
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

std::vector<double> reluVector()
