#include <cerrno>
#include <iterator>
#include <random>
#include <vector>
class qLearning {
public:
  int rows, cols;
  std::vector<std::vector<std::vector<double>>> qTabular;

  double alpha = 0.2;
  double gamma = 0.99;
  double epsilon = 1.0;
  double epsilonDecay = 0.999;
  double miniEpsilon = 0.05;

  std::mt19937 rng;
  qLearning(int rows, int cols) {
    std::random_device rd;
    rng = std::mt19937(rd());

    qTabular.resize(rows, std::vector<std::vector<double>>(
                              cols, std::vector<double>(4, 0.0)));
  }
  int pickAction(int ii, int jj) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    if (distribution(rng) < epsilon) {
      std::uniform_int_distribution<int> actionDistribution(0, 3);
      int action = actionDistribution(rng);
      return action;
    } else {
      int bestAction;
      int maxValue = -999999;

      for (int i = 0; i < 4; i++) {
        if (qTabular[ii][jj][i] > maxValue) {
          maxValue = qTabular[ii][jj][i];
          bestAction = i;
        }
      }
      return bestAction;
    }
  }

  void learn(int currentI, int currentJ, int nextI, int nextJ, double reward,
             int action) {
    double previousQ = qTabular[currentI][currentJ][action];
    double maxValue = -99999;
    for (int i = 0; i < 4; i++) {
      if (qTabular[nextI][nextJ][i] > maxValue)
        maxValue = qTabular[nextI][nextJ][i];

      double newQ = previousQ + alpha * (reward + gamma * maxValue - previousQ);
      qTabular[nextI][nextJ][action] = newQ;
    }
  }

  void epsilonDelta() {
    if (epsilon > miniEpsilon)
      epsilon *= epsilonDecay;
  }
};
