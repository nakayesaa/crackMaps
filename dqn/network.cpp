#include "../dqn/layer.cpp"
#include <vector>

std::vector<double> vectorSubss(const std::vector<double> &vectorA,
                                const std::vector<double> &vectorB) {
  std::vector<double> output;
  for (int i = 0; i < vectorA.size(); i++) {
    output[i] = vectorA[i] - vectorB[i];
  }
  return output;
}

class Network {
private:
  // stack of layer
  std::vector<Layer *> layers;

public:
  ~Network() {
    for (auto layer : layers)
      delete layer;
  }

  void add(Layer *layer) { layers.push_back(layer); }

  Network *clone() {
    Network *network = new Network();
    for (auto layer : layers) {
      network->add(layer->clone());
    }
    return network;
  }

  std::vector<double> forwardPass(std::vector<double> input) {
    for (auto layer : layers) {
      input = layer->forward(input);
    }
    return input;
  }

  void train(std::vector<double> input, std::vector<double> target,
             double learningRate) {
    std::vector<double> output = forwardPass(input);
    std::vector<double> error = vectorSubs(output, target);

    for (int i = layers.size() - 1; i >= 0; i--) {
      error = layers[i]->backward(error, learningRate);
    }
  }
};
