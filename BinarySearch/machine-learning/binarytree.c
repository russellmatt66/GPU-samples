#include <stdio.h>
#include <stdlib.h>

// Library functions for implementing Binary Tree
typedef struct BTNode{
    int val;
    int depth; 
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

// Free 'delNode'
void freeNode(BTNode* delNode){
    if (delNode != NULL){
        free(delNode);
    }
    return;
}

// Free the entire memory of the tree 
void freeBT(BTNode* root){
    if (root != NULL){
        freeBT(root->left);
        freeBT(root->right);
    }
    freeNode(root);
    return;
}