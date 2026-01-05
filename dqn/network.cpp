#include "../dqn/layer.cpp"
#include <vector>
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
};
