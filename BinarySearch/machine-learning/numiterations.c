// Code to calculate the number of iterations that it takes to find a particle on average using a binary tree
#include "binarytree.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>

// THERE IS A SLIGHT FLAW IN THIS LOGIC
// THE FLAW IS THAT I DID NOT UNIFORMLY INITIALIZE THE PARTICLES
// I RANDOMLY INITIALIZED THEM (UNIFORM DISTRIBUTION)
// The solution is to randomly select cells 'N' times, and obtain statistics. Honestly, easy enough problem, can even use CUDA for it! 
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

int main(int argc, char* argv[]){
    int Nx = atoi(argv[1]); // need to be able to pass in the number of gridpoints
    int N = atoi(argv[2]); // need to know how many particles

    // printf("%d\n", Nx);
    int level = 1;
    double avg_iter; 

    // Create binary tree
    int low = 0, high = Nx-1;
    int guess = (low + high) / 2;
    BTNode* root = createBTNode(guess, level);
    buildLeaves(root, Nx, low, high, guess, level + 1);
    // printNode(root);
    
    // Calculate average number of iterations
    int total_iterations = 0;

    int *p_cells, *cells, *num_node;
    
    p_cells = (int*)malloc(N*sizeof(int));
    cells = (int*)malloc((Nx-1)*sizeof(int));
    num_node = (int*)malloc(sizeof(int));

    *num_node = 0;
    
    getCells(root, cells, num_node); // Initializes cells

    srand(time(NULL));
    int cell = rand() % (Nx - 1);
    
    for (int i = 0; i < N; i++){
        p_cells[i] = cell;
        cell = rand() % (Nx - 1);
        total_iterations += cells[p_cells[i]];
    }

    // printf("%d\n", total_iterations);

    avg_iter = (double)total_iterations / N;
    // printf("It takes %f iterations on average to find a particle\n", avg_iter); 

    freeBT(root);
    free(p_cells);
    free(cells);
    free(num_node);

    printf("%f\n", avg_iter);
    return 0;
}