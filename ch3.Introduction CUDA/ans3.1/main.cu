#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
__global__ void a(float *a,float * b,float *c,size_t n) {
    int i=blockIdx.y*blockDim.x+blockIdx.x;
    if(i > n*n) return;
    a[i]=b[i]+c[i];
}


__global__ void b(float *a,float * b,float *c,size_t n) {
    int offset=blockIdx.y*blockDim.x+blockIdx.x;
    if(offset > n) return;
    for(int i=0;i<dim;++i){
        a[i+offset]=b[i+offset]+c[i+offset];
    }
}


__global__ void c(float *a,float * b,float *c,size_t n) {
    int offset=blockIdx.y*blockDim.x+blockIdx.x;
    if(offset > n) return;
    for(int i=0;i<dim;++i){
        a[i*dim+offset]=b[i*dim+offset]+c[i*dim+offset];
    }
}

void stub(){
    size_t dim=100;
    size_t siz=dim*dim*sizeof(float);
    float* B=(float*)malloc(siz);
    float* A=(float*)malloc(siz);
    float* C=(float*)malloc(siz);
    float* c_A,c_B,c_C;
    cudaMalloc((void**)&c_A, siz);
    cudaMalloc((void**)&c_B, siz);
    cudaMalloc((void**)&c_C, siz);
    a<<<ceil( dim*dim / 256),256>>>(c_A,c_B,c_C,dim);
    b<<<ceil(dim / 256),256>>>(c_A,c_B,c_C,dim);
    c<<<ceil(dim / 256),256>>>(c_A,c_B,c_C,dim);
    
    cudaFree(c_A);
    cudaFree(c_B);
    cudaFree(c_C);
}

int main(){
    stub();
    return 0;
}