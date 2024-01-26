#ifndef PARTICLES_HPP
#define PARTICLES_HPP

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

struct Point{
    float x, y, z;

    __host__ __device__ Point(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    __host__ __device__ void print() const {
        printf("Point: (%.2f, %.2f, %.2f)\n", x, y, z);
    }
};

struct ChargedParticle{
    Point coord;
    float q;

    __host__ __device__ ChargedParticle(Point _P, float _q): coord(_P), q(_q) {}

    __host__ __device__ void print() const {
        coord.print();
        printf("Charge: %.2f\n", q);
    }
};

#endif