#include "algorithm"
#include "deque"
#include "random"
#include "vector"
#include <algorithm>
#include <cstddef>
#include <deque>
#include <iterator>
#include <random>
#include <vector>

struct buffer {
  int action;
  double reward;
  std::vector<double> currentState;
  std::vector<double> nextState;
  bool isDone;
};

class replayBuffer {
public:
  std::deque<buffer> Buffer;
  std::size_t maxSize;

  replayBuffer(std::size_t size) : maxSize(size) {}

  void add(buffer buff) {
    Buffer.push_back(buff);
    if (Buffer.size() > maxSize)
      Buffer.pop_front();
  }

  std::vector<buffer> sample(int batchSize) {
    std::vector<buffer> batch;
    std::sample(Buffer.begin(), Buffer.end(), std::back_inserter(batch),
                batchSize, std::mt19937{std::random_device{}()});
    return batch;
  }

  std::size_t size() { return Buffer.size(); }
};
