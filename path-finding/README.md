# Intelligent Systems - "Help the Robot" Pathfinding Project

## Overview

This project involves developing an intelligent agent that navigates a topographic map to find an optimal path from a starting point to a destination using various search algorithms. The robot navigates based on terrain types and movement costs.

## Problem Description

### The Task
Develop an intelligent agent to navigate an NxN topographic map consisting of normal ground (`R`), cliffs (`X`), a starting point (`S`), and a goal point (`G`).

### Movement Rules
1. The robot can move to any of the eight adjacent cells.
2. The robot cannot move into cliff cells (`X`).
3. Diagonal moves are prohibited if adjacent cells are cliffs.
4. Movement costs:
  - **Diagonal Moves**: Cost = 1
  - **Up, Down, Left, Right**: Cost = 2

### Input Format
- **File**: `input.txt`
  - Line 1: Algorithm to use (1 = BFS, 2 = UCS, 3 = IDS, 4 = A*, 5 = Hill Climbing)
  - Line 2: Board size (`N`)
  - Remaining Lines: Board layout (`R`, `X`, `S`, `G`)

### Output Format
- **File**: `output.txt`
  - Solution path (e.g., `R-R-DR-D-L-DL`)
  - Total path cost
  - Path length
  - Number of explored states (or steps for Hill Climbing)
  - Visual representation of the path
  - Any additional statistics

## Algorithms to Implement

1. **Breadth-First Search (BFS)**
2. **Uniform Cost Search (UCS)**
3. **Iterative Deepening Search (IDS)**
4. **A* Search**
5. **Hill Climbing** (three variations):
  - Greedy Hill Climbing (Steepest Ascent)
  - Random Restart Hill Climbing
  - Randomised Hill Climbing

## Implementation Requirements

### Code
- Implement all algorithms in Python.
- Read input from `input.txt` and output results to `output.txt`.
- Ensure code is well-documented with comments explaining key functions and logic.

### Experiments
- Test the program on various board configurations.
- Compare performance metrics: solution quality, number of explored states, and heuristic impact.
- Handle cases where no path exists between the start and goal.