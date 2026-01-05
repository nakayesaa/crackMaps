#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

double dotProduct(const std::vector<double> &vectorA,
                  const std::vector<double> &vectorB) {
  double finalResult = 0;
  for (std::size_t i = 0; i < vectorA.size(); i++) {
    finalResult += vectorA[i] * vectorB[i];
  }
  return finalResult;
}

std::vector<double> elementWise(std::vector<double> vector, int scalar) {
  for (std::size_t i = 0; i < vector.size(); i++) {
    vector[i] *= scalar;
  }
  return vector;
}

std::vector<double> vectorSubs(const std::vector<double> &vectorA,
                               const std::vector<double> &vectorB) {
  std::vector<double> finalResult(vectorA.size());
  for (std::size_t i = 0; i < vectorA.size(); i++) {
    finalResult[i] = vectorA[i] - vectorB[i];
  }
  return finalResult;
}

std::vector<std::vector<double>>
transpose(const std::vector<std::vector<double>> &matrix) {
  int rows = matrix.size();
  int cols = matrix[0].size();

  std::vector<std::vector<double>> finalResult(cols, std::vector<double>(rows));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      finalResult[j][i] = matrix[i][j];
    }
  }
  return finalResult;
}

std::vector<std::vector<double>> assignWeigth(int rows, int cols) {
  static std::random_device rd;
  static std::mt19937 generateRandom(rd());

  std::vector<std::vector<double>> weigth(rows, std::vector<double>(cols));
  double bound = std::sqrt(6.0 / (rows + cols));
  std::uniform_real_distribution<double> interval(-bound, bound);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      weigth[i][j] = interval(generateRandom);
    }
  }
  return weigth;
}
