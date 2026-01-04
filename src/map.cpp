#include "../include/map.h"
#include "queue"
#include "random"
#include "vector"

std::vector<std::vector<int>> backtrackMaze(int width, int height) {
  if (width % 2 == 0)
    width++;
  if (height % 2 == 0)
    height++;

  std::vector<std::vector<int>> maze(height, std::vector<int>(width, 1));

  std::random_device random;
  std::mt19937 generateRandomness(random());

  int firstRow = (std::uniform_int_distribution<int>(0, (height - 1) / 2)(
                     generateRandomness)) *
                     2 +
                 1;
  int firstColumn = (std::uniform_int_distribution<int>(0, (width - 1) / 2)(
                        generateRandomness)) *
                        2 +
                    1;

  maze[firstRow][firstColumn] = 0;
  std::vector<std::vector<int>> stack;
  stack.push_back({firstRow, firstColumn});

  while (!stack.empty()) {
    std::vector<int> position = stack.back();
    int row = position[0];
    int col = position[1];

    std::vector<std::vector<int>> neighbors = checkNeighbors(maze, row, col);
    if (neighbors.empty()) {
      stack.pop_back();
    } else {
      std::uniform_int_distribution<int> interval(0, neighbors.size() - 1);
      std::vector<int> pick = neighbors[interval(generateRandomness)];

      maze[pick[0]][pick[1]] = 0;

      int betweenRow = (row + pick[0]) / 2;
      int betweenCol = (col + pick[1]) / 2;

      maze[betweenRow][betweenCol] = 0;
      stack.push_back(pick);
    }
  }
  return maze;
}

std::vector<std::vector<int>>
checkNeighbors(const std::vector<std::vector<int>> &maze, int cRow, int cCol) {
  std::vector<std::vector<int>> neighbors;

  int rowDirection[4] = {-2, 2, 0, 0};
  int colDirection[4] = {0, 0, 2, -2};

  for (int i = 0; i < 4; i++) {
    int row = cRow + rowDirection[i];
    int col = cCol + colDirection[i];

    if (row < maze.size() && col < maze[row].size() && row >= 0 && col >= 0) {
      if (maze[row][col] == 1)
        neighbors.push_back({row, col});
    }
  }
  return neighbors;
}
