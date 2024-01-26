#ifndef GRID_HPP
#define GRID_HPP

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include "Particles.hpp"

struct CartesianGrid{
    Point*** grid;
    int Nx, Ny, Nz;
    float x_min, x_max, y_min, y_max, z_min, z_max;
    
    __host__ __device__ CartesianGrid(int _Nx, int _Ny, int _Nz, float _xmin, float _xmax, float _ymin, float _ymax, float _zmin, float _zmax) :
    Nx(_Nx), Ny(_Ny), Nz(_Nz), x_min(_xmin), x_max(_xmax), y_min(_ymin), y_max(_ymax), z_min(_zmin), z_max(_zmax) 
    {
        grid = (Point***)malloc(_Nx * _Ny * _Nz * sizeof(Point));
        // Initialize the points inside the grid
    }
};

#endif