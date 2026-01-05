#include "../dqn/network.cpp"
#include "../dqn/replay.cpp"
#include <random>
#include <vector>

class dqnAgent {
public:
  Network *Policy;
  Network *Target;
  replay buffer;

  double gamma = 0.95;
  double epsilon = 1.0;
  double minimumEpsilon = 0.01;
  double epsilonDecay = 0.995;
  double learningRate = 0.01;

  int actionSize;
  std::mt19937 generateRandom;

  dqnAgent(int inputSize, int outputSize)
      : buffer(2000), actionSize(outputSize) {
    Policy = new Network();
    Policy->add(new linearLayer(inputSize, 64));
    Policy->add(new reluLayer());
    Policy->add(new linearLayer(64, outputSize));

    Target = Policy->clone();
    std::random_device rd;
    generateRandom = std::mt19937(rd());
  }

  ~dqnAgent() {
    delete Policy;
    delete Target;
  }

  int doAction(const std::vector<double> state) {
    std::uniform_real_distribution<double> distributionNumber(0.0, 1.0);

    if (distributionNumber(generateRandom) < epsilon) {
      std::uniform_int_distribution<int> actionDistribution(0, actionSize - 1);
      return actionDistribution(generateRandom);
    }

    std::vector<double> qValue = Policy->forwardPass(state);
    int bestAction = 0;

    for (int i = 1; i < qValue.size(); i++) {
      if (qValue[i] > qValue[bestAction]) {
        bestAction = i;
      }
    }
    return bestAction;
  }

  void train(int batchSize) {
    if (buffer.size() < batchSize)
      return;

    std::vector<Experience> batch = buffer.sample(batchSize);

    for (auto &experience : batch) {
      std::vector<double> currentQ =
          Policy->forwardPass(experience.currentState);

      double targetQ = experience.reward;
      if (!experience.isDone) {
        std::vector<double> nextQ = Target->forwardPass(experience.nextState);
        double maxNextQ = nextQ[0];
        for (double value : nextQ) {
          if (value > maxNextQ)
            maxNextQ = value;
        }
        targetQ += gamma * maxNextQ;
      }
      std::vector<double> targetVector = currentQ;
      targetVector[experience.action] = targetQ;

      Policy->train(experience.currentState, targetVector, learningRate);
    }

    if (epsilon > minimumEpsilon) {
      epsilon *= epsilonDecay;
    }
  }

  void updateTargetNetwork() {
    delete Target;
    Target = Policy->clone();
  }
};
