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

// Add 'newNode' to 'parent', which connection to make depends on 'direction' 
void addNode(BTNode* parent, BTNode* newNode, char* direction){
    if (strcmp(direction, "left") == 0){
        parent->left = newNode;
    }
    else if (strcmp(direction, "right") == 0){
        parent->left = newNode;
    }
    return;
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