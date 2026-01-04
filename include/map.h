#pragma once
#include <vector>
std::vector<std::vector<int>> backtrackMaze(int width, int heigth);
std::vector<std::vector<int>>
checkNeighbors(const std::vector<std::vector<int>> &maze, int i, int j);
