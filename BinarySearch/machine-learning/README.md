# Overview
For large N, and Nx, i.e., problems approaching the limits of the GTX 960's capabilities, the CUDA-based timer began to malfunction, producing nonsense. Consequentially, to get a sense of the performance of the code in these regimes, machine learning (ML) models based on the Python scikit-learn (sklearn) library are built, and trained on the valid data.

Beyond this, the goal of this part of the project is also to analyze the datasets for which accurate timing information was obtained.  

# Directory Structure
gtx960-kerneldata/
- Raw, benchmarking data for runs performed on a GeForce GTX 960
- Contains some malformed data, but no missing values

gtx960-cleandata/
- Contains .csv containing statistics for all the clean datasets
- Contains .txt listing all the dirty datasets

clean.py
- Script that operates on a '*-kerneldata/' folder which contains datasets from a benchmarking run on a GPU 
- Produces a directory, '*-clean/', which contains the clean datasets, and a list of the malformed ones

binarytree.c/
- Library functions for instantiating a binary tree

numiterations.c/
- Use 'binarytree.c' to calculate correct value for 'avg_iters' to put into effective bandwidth formula

analyze.py
- Script that operates on a '*-clean/' directory, and computes a number of relevant performance metrics
- Also visualizes the performance landscapes of the various problems

randomforest.py
- Code that builds, trains, and deploys ML models on datasets from a '*-clean/' directory.
- Purpose of the models, which are based on the Random Forest architecture, is to predict the performance of the code on the problem sizes which were so large that the CUDA-based timer malfunctioned. 

project.gv
- Graphviz source code for visualizing the structure of the project
