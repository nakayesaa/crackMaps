#include "../include/agent.h"
#include "../include/map.h"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <limits>
#include <vector>

int main() {
  int size = 20;
  int cols = 30;
  int rows = 30;

  std::vector<std::vector<int>> maze = backtrackMaze(cols, rows);
  int height = maze.size();
  int width = maze[0].size();

  qLearning agent(height, width);

  int currentY = 1;
  int currentX = 1;
  int goalY = height - 2;
  int goalX = width - 2;

  int directionY[4] = {-1, 1, 0, 0};
  int directionX[4] = {0, 0, -1, 1};

  sf::RenderWindow window(sf::VideoMode(width * size, height * size),
                          "RL Maze Solver");
  window.setFramerateLimit(60);

  int episode = 0;
  int currentSteps = 0;
  int bestSteps = std::numeric_limits<int>::max();
  int stepsPerFrame = 200;

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window.close();
    }

    for (int k = 0; k < stepsPerFrame; k++) {
      int action = agent.pickAction(currentY, currentX);
      int nextY = currentY + directionY[action];
      int nextX = currentX + directionX[action];

      double reward = -0.1;
      bool hitWall = false;
      bool hitGoal = false;
      bool terminate = false;

      int maxAttempt = (bestSteps == std::numeric_limits<int>::max())
                           ? (width * height * 5)
                           : (bestSteps * 2);

      if (currentSteps > maxAttempt) {
        terminate = true;
      }

      if (nextY < 0 || nextY >= height || nextX < 0 || nextX >= width) {
        reward = -10.0;
        hitWall = true;
      } else if (maze[nextY][nextX] == 1) {
        reward = -5.0;
        hitWall = true;
      } else if (nextY == goalY && nextX == goalX) {
        reward = 100.0;
        hitGoal = true;
      }

      int pickNextY = hitWall ? currentY : nextY;
      int pickNextX = hitWall ? currentX : nextX;
      agent.learn(currentY, currentX, pickNextY, pickNextX, reward, action);

      if (!hitWall) {
        currentY = nextY;
        currentX = nextX;
        currentSteps++;
      }
      if (hitGoal || terminate) {
        episode++;

        if (hitGoal) {
          std::cout << "episode: " << episode << " : " << currentSteps
                    << " steps" << std::endl;
          if (currentSteps < bestSteps) {
            if (bestSteps != std::numeric_limits<int>::max()) {
              double percentageGain =
                  ((double)(bestSteps - currentSteps) / bestSteps) * 100.0;
              std::cout << "new best : episode " << episode << " : "
                        << currentSteps << " steps : " << percentageGain
                        << "% better" << std::endl;
            }
            bestSteps = currentSteps;
          }
        } else if (terminate)
          agent.learn(currentY, currentX, currentY, currentX, -10.0, action);

        currentY = 1;
        currentX = 1;
        currentSteps = 0;
        agent.epsilonDelta();

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
