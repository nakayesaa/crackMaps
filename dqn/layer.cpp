#include "../dqn/helper.cpp"
#include <vector>

class Layer {
public:
  std::vector<double> input;
  std::vector<double> output;

  virtual std::vector<double> forward(const std::vector<double> &inputData) = 0;
  virtual std::vector<double> backward(const std::vector<double> &inputError,
                                       double learningRate) = 0;
  virtual Layer *clone() = 0;
};

class linearLayer : public Layer {
public:
  int inputLayer;
  int outputLayer;
  std::vector<std::vector<double>> weigth;
  std::vector<double> bias;

  linearLayer(int input, int output) {
    inputLayer = input;
    outputLayer = output;
    weigth = assignWeigth(output, input);
    bias.assign(output, 0);
  }

  linearLayer(const linearLayer &copy) {
    inputLayer = copy.inputLayer;
    outputLayer = copy.outputLayer;
    weigth = copy.weigth;
    bias = copy.bias;
  }

  Layer *clone() override {
    // its just copy
    return new linearLayer(*this);
  }

  std::vector<double> forward(const std::vector<double> &inputData) override {
    input = inputData;
    output.clear();
    output.reserve(outputLayer);

    for (int i = 0; i < outputLayer; i++) {
      // the goat y = wx + b
      output.push_back(dotProduct(weigth[i], input) + bias[i]);
    }

    return output;
  }

  std::vector<double> backward(const std::vector<double> &inputError,
                               double learningRate) override {
    std::vector<double> error(inputLayer, 0.0);

    for (int i = 0; i < outputLayer; i++) {
      for (int j = 0; j < inputLayer; j++) {
        // basically matrixTranspose * error for prev layer
        error[j] += weigth[i][j] * inputError[i];
      }
    }

    for (int i = 0; i < outputLayer; i++) {
      // update bias
      bias[i] -= (learningRate * inputError[i]);

      for (int j = 0; j < inputLayer; j++) {
        double gradient = inputError[i] * input[j];
        weigth[i][j] -= (gradient * learningRate);
      }
    }

    return error;
  }
};
