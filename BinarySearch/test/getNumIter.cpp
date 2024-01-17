#include <cmath>
#include <string>
#include <iostream>

// Compute the number of iterations associated with node i
int getNumIter(int i){
    // This is an easy problem
    int num_iter = 1;
    int k = 0;
    while (k < i){
        k += pow(2, num_iter);
        num_iter++;
    }
    return num_iter;
}

int main(int argc, char* argv[]){
    int Nx = std::stoi(argv[1]);
    int num_iter = 0;
    for (int i = 0; i < Nx-1; i++){
        num_iter = getNumIter(i);
        std::cout << "Node " << i << " is associated with iteration " << num_iter << std::endl;
    }
    return 0;
}