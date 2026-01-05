;
#include <algorithm>
#include <deque>
#include <iterator>
#include <random>
#include <vector>

struct Experience {
  int action;
  double reward;
  std::vector<double> currentState;
  std::vector<double> nextState;
  bool isDone;
};

class replay {
public:
  std::deque<Experience> replayBuffer;
  std::size_t maxSize;

  replay(std::size_t size) : maxSize(size) {}

  void add(Experience buff) {
    replayBuffer.push_back(buff);
    if (replayBuffer.size() > maxSize)
      replayBuffer.pop_front();
  }

  std::vector<Experience> sample(int batchSize) {
    std::vector<Experience> batch;
    std::sample(replayBuffer.begin(), replayBuffer.end(),
                std::back_inserter(batch), batchSize,
                std::mt19937{std::random_device{}()});
    return batch;
  }

  std::size_t size() { return replayBuffer.size(); }
};
