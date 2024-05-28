#ifndef MATRIX_CU
#define MATRIX_CU
/*
Header file that implements Matrix class
It seems with CUDA, the best approach is to just do stuff with a 1D pointer
*/

// Too complex
// class Matrix{
// 	// TODO
//     private:
//         int rows_;
//         int cols_;
//         float* data_;
    
//     public:
//         __host__ __device__ Matrix(int rows, int cols) : rows_(rows), cols_(cols)
//         {
//             data_ = new float[rows * cols];
//         }

//         __host__ __device__ ~Matrix() {
//             delete[] data_;
//         }

//         // row-major order
//         __host__ __device__ const float getElement(const int i, const int j){
//             return data_[cols_ * i + j]; 
//         }

//         __host__ __device__ void setElement(const float val, const int i, int j){
//             data_[cols_ * i + j] = val;
//         }
// };
#endif