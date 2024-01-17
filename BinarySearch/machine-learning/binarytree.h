#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Library functions for implementing Binary Tree
typedef struct BTNode{
    int val; // the cell where the search is looking 
    int depth; // number of iterations to get here
    struct BTNode* left;
    struct BTNode* right;
} BTNode;

BTNode* createBTNode(int value, int depth) {
    BTNode* newNode = (BTNode*)malloc(sizeof(BTNode));
    if (newNode != NULL) { 
        newNode->val = value;
        newNode->depth = depth;
        newNode->left = NULL;
        newNode->right = NULL;
    } else {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }
    return newNode;
}

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

int sumIters(BTNode* root, int sum){
    if (root == NULL){
        return 0;
    }
    sum = root->depth; // the level of a node corresponds to how many iterations it takes binary search to find the particle in the associated grid-cell
    return sum + sumIters(root->left, sum) + sumIters(root->right, sum);
}

void printNode(BTNode* root){
    // Traverse tree and print the values and depths out
    if (root == NULL){
        return;
    }
    printf("Algorithm finds particle in grid-cell %d, in %d iterations\n", root->val, root->depth);
    printNode(root->left);
    printNode(root->right);
}

// Free 'delNode'
void freeNode(BTNode* delNode){
    if (delNode == NULL){
        return;
    }
    free(delNode);
}

// Free the entire memory of the tree 
void freeBT(BTNode* root){
    if (root == NULL){
        return;
    }
        freeBT(root->left);
        freeBT(root->right);
        freeNode(root);
}