// Code to calculate the number of iterations that it takes to find a particle on average using a binary tree
#include "bst.c"

// Logic:
// (I)
// Create a binary tree where the values of the nodes correspond to the index of a grid-cell in a uniform grid, 
// where a binary search algorithm is searching for a population of uniformly-spaced particles. 
// (II)
// Construct the tree so that the root contains the value of the midpoint grid-cell, where the search algorithm begins.
// Subsequently, add nodes to the left, and right, of a parent node such that their values represent the
// index of the grid-cell that the next iteration of the algorithm would be looking for the particle in, if it detected that the particle was to
// the left, or right, of the grid-cell associated with the parent.
// EXAMPLE:
// Nx = 16, low = 0, high = Nx-1 = 15
// \therefore (initial) guess = floor((0 + 15) / 2) = 7  
// parent = root, root->val = (initial) guess = 7 (grid-cell where we are guessing the particle is initially)
// Binary search looks for particle in grid[guess=7]
// IF it determines that the particle is TO THE LEFT of the grid-cell: particle.pos < grid[guess]
// THEN high = guess, and the new guess = floor((l+h)/2) = floor((0+7)/2) = 3 
// SO parent->left->val = 3
// IF it determines that the particle is TO THE RIGHT of the grid-cell: particle.pos > grid[guess]
// THEN low = guess, and the new guess = floor((l+h)/2) = floor((7+15)/2) = 11
// SO parent->right->val = 11
// (III)
// While constructing the tree in this way, the BTNode->depth value will be used to hold the number of iterations that it takes to find a particle
// inside the grid-cell identified by BTNode->val. The total number of iterations can be calculated by traversing the tree and summing up all the  
// BTNode->depth values, and the average number of iterations that it takes to find a particle is then found by dividing that number by the number of 
// grid-cells.