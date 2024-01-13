// Code to calculate the number of iterations that it takes to find a particle on average using a binary tree
#include "binarytree.c"
#include <math.h>

BTNode* buildTree();

// Logic:
// (I)
// Create a binary tree with the values of the nodes corresponding to the index of a grid-cell in a uniform grid, where a binary search algorithm 
// is searching for a population of uniformly-spaced particles. 
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
// inside the grid-cell identified by BTNode->val. For a population of uniformly-spaced particles, the total number of iterations can be calculated 
// by traversing the tree and summing up all the BTNode->depth values, and the average number of iterations that it takes to find a particle is then 
// found by dividing that number by the number of grid-cells.

void buildLeaves(BTNode* parent, int Nx, int low, int high, int guess, int level){
    if (parent == NULL || level > (int)log2(Nx)){
        return;
    }
    int left_low = low; 
    int left_high = guess;
    int left_guess = (left_low + left_high) / 2;
    int right_low = guess;
    int right_high = high;
    int right_guess = (right_low + right_high) / 2;
    BTNode* leftNode = createBTNode(left_guess, level);
    BTNode* rightNode = createBTNode(right_guess, level);
    parent->left = leftNode;
    parent->right = rightNode;
    buildLeaves(parent->left, Nx, left_low, left_high, left_guess, level + 1);
    buildLeaves(parent->right, Nx, right_low, right_high, right_guess, level + 1);
}

int main(int argc, char* argv[]){
    int Nx = atoi(argv[1]); // need to be able to pass in the number of gridpoints
    int avg_iter, level = 1; 
    // printf("%d\n", Nx);

    // Create binary tree
    int low = 0, high = Nx-1;
    int guess = (low + high) / 2;
    BTNode* root = createBTNode(guess, level);
    buildLeaves(root, Nx, low, high, guess, level + 1);
    printNode(root);
    freeBT(root);
    return 0;
    // return avg_iter;
}