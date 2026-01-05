#include "../dqn/dqnAgent.cpp"
#include "../include/map.h"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <limits>
#include <vector>

std::vector<double> getFlattenedState(const std::vector<std::vector<int>> &maze,
                                      int agentY, int agentX) {
  std::vector<double> state;
  int h = maze.size();
  int w = maze[0].size();
  state.reserve(h * w);

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (y == agentY && x == agentX) {
        state.push_back(1.0);
      } else if (maze[y][x] == 1) {
        state.push_back(-1.0);
      } else {
        state.push_back(0.0);
      }
    }
  }
  return state;
}

int main() {
  int size = 20;
  int cols = 15;
  int rows = 15;

  std::vector<std::vector<int>> maze = backtrackMaze(cols, rows);
  int height = maze.size();
  int width = maze[0].size();

  dqnAgent agent(height * width, 4);

  int currentY = 1;
  int currentX = 1;
  int goalY = height - 2;
  int goalX = width - 2;

  int directionY[4] = {-1, 1, 0, 0};
  int directionX[4] = {0, 0, -1, 1};

  sf::RenderWindow window(sf::VideoMode(width * size, height * size),
                          "Deep Q-Network Maze Solver");
  window.setFramerateLimit(60);

  int episode = 0;
  int currentSteps = 0;
  int totalSteps = 0;
  int bestSteps = std::numeric_limits<int>::max();

  int stepsPerFrame = 50;
  int batchSize = 32;

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window.close();
    }

    for (int k = 0; k < stepsPerFrame; k++) {
      std::vector<double> state = getFlattenedState(maze, currentY, currentX);

      int action = agent.doAction(state);
      int nextY = currentY + directionY[action];
      int nextX = currentX + directionX[action];

      double reward = -0.1;
      bool hitWall = false;
      bool hitGoal = false;
      bool terminate = false;

      int maxAttempt = (bestSteps == std::numeric_limits<int>::max())
                           ? (width * height * 2)
                           : (bestSteps * 3);
      if (currentSteps > maxAttempt)
        terminate = true;

      if (nextY < 0 || nextY >= height || nextX < 0 || nextX >= width) {
        reward = -1.0;
        hitWall = true;
      } else if (maze[nextY][nextX] == 1) {
        reward = -1.0;
        hitWall = true;
      } else if (nextY == goalY && nextX == goalX) {
        reward = 10.0;
        hitGoal = true;
      }

      int pickNextY = hitWall ? currentY : nextY;
      int pickNextX = hitWall ? currentX : nextX;

      std::vector<double> nextState =
          getFlattenedState(maze, pickNextY, pickNextX);
      bool done = hitGoal || terminate;

      Experience exp = {action, reward, state, nextState, done};
      agent.buffer.add(exp);

      agent.train(batchSize);

      totalSteps++;
      if (totalSteps % 1000 == 0) {
        agent.updateTargetNetwork();
      }

      if (!hitWall) {
        currentY = nextY;
        currentX = nextX;
      }
      currentSteps++;

      if (done) {
        episode++;
        if (hitGoal) {
          std::cout << "Episode: " << episode << " | Steps: " << currentSteps
                    << " | Epsilon: " << agent.epsilon << std::endl;

          if (currentSteps < bestSteps) {
            bestSteps = currentSteps;
            std::cout << ">>> NEW BEST! " << bestSteps << " steps."
                      << std::endl;
          }
        }
        currentY = 1;
        currentX = 1;
        currentSteps = 0;
        break;
      }
    }

    window.clear(sf::Color(50, 50, 50));
    sf::RectangleShape shape(sf::Vector2f(size, size));

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        shape.setPosition(x * size, y * size);
        shape.setFillColor(maze[y][x] == 1 ? sf::Color::Black
                                           : sf::Color::White);
        window.draw(shape);
      }
    }

    shape.setPosition(goalX * size, goalY * size);
    shape.setFillColor(sf::Color::Green);
    window.draw(shape);

    shape.setPosition(currentX * size, currentY * size);
    shape.setFillColor(sf::Color::Red);
    shape.setSize(sf::Vector2f(size * 0.8f, size * 0.8f));
    shape.move(size * 0.1f, size * 0.1f);
    window.draw(shape);

    window.display();
  }
  return 0;
}
