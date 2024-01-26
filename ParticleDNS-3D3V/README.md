# Overview
This project aims to implement a CUDA simulation of a population of charged particles, one which directly evolves the motion of each of the particles using expressions for the Lienard-Wiechert fields, ala a gravitational N-body simulation.  

The goal of this 3D3V simulation is to study the effect that a fusion alpha has on electrons in its vicinity. 

# Directory Structure


# Current Tasks
(1) Implement a modular core
- The kernel will go through a list of particles, and for each particle, calculate the electromagnetic force on it from the other particles, based on the Lienard-Wiechert fields. 
    - How should these particles be represented?