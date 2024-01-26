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

__global__ void pushLienardWiechert(){
    return;
}
int main(int argc, char* argv[]){
    int N = atoi(argv[1]);
    int Nx = atoi(argv[2]);
    int Ny = atoi(argv[3]);
    int Nz = atoi(argv[4]);
    return 0;
}