#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "../../comm/helper_cuda.h"

__global__ void a(float *a,float * b,float *c,size_t n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i > n*n) return;
    a[i]=b[i]+c[i];
}


__global__ void b(float *a,float * b,float *c,size_t n) {
    int offset=blockIdx.x*blockDim.x+threadIdx.x;
    if(offset > n) return;
    for(int i=0;i<n;++i){
        a[i+offset]=b[i+offset]+c[i+offset];
    }
}


__global__ void c(float *a,float * b,float *c,size_t n) {
    int offset=blockIdx.x*blockDim.x+threadIdx.x;
    if(offset > n) return;
    for(int i=0;i<n;++i){
        a[i*n+offset]=b[i*n+offset]+c[i*n+offset];
    }
}
template<typename T>
void rand_array(T * array,size_t len){
    for(int i=0;i<len;++i){
        array[i]=((T)rand())/RAND_MAX;
    }
}
bool check(float* A,float* B,float * C,size_t dim){
    float* tmp=(float*)malloc(dim*dim*sizeof(float));
    for(int i=0;i<dim;++i){
        for(int j=0;j<dim;++j){
            tmp[i*dim+j]=B[i*dim+j]+C[i*dim+j];
            if(fabs(tmp[i*dim+j]-A[i*dim+j])>1e-10){ 
                free(tmp);
                return false;
            }
        }
    }
    free(tmp);
    return true;
}
int stub(){
    size_t dim=100;
    size_t siz=dim*dim*sizeof(float);
    float* B=(float*)malloc(siz);
    float* A=(float*)malloc(siz);
    float* C=(float*)malloc(siz);
    if(B==nullptr || A==nullptr || C==nullptr)
        return -1;
    float* c_A,*c_B,*c_C;
    rand_array(B,dim);
    rand_array(C,dim);    
    checkCudaErrors(cudaMalloc((void**)&c_A, siz));
    checkCudaErrors(cudaMalloc((void**)&c_B, siz));
    checkCudaErrors(cudaMalloc((void**)&c_C, siz));
    checkCudaErrors(cudaMemcpy(c_B, B, siz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(c_C, C, siz, cudaMemcpyHostToDevice));
    a<<<ceil( dim*dim / 256),256>>>(c_A,c_B,c_C,dim);
    checkCudaErrors(cudaMemcpy(A, c_A, siz, cudaMemcpyDeviceToHost));
    if(!check(A,B,C,dim)){
        std::cout<<"A kernal err\n";
        return -1;
    } 
    b<<<ceil(dim / 256),256>>>(c_A,c_B,c_C,dim);
    checkCudaErrors(cudaMemcpy(A, c_A, siz, cudaMemcpyDeviceToHost));
    if(!check(A,B,C,dim)){
        std::cout<<"B kernal err\n";
        return -1;
    } 
    c<<<ceil(dim / 256),256>>>(c_A,c_B,c_C,dim);
    checkCudaErrors(cudaMemcpy(A, c_A, siz, cudaMemcpyDeviceToHost));
    if(!check(A,B,C,dim)){
        std::cout<<"C kernal err\n";
        return -1;
    } 
    checkCudaErrors(cudaFree(c_A));
    checkCudaErrors(cudaFree(c_B));
    checkCudaErrors(cudaFree(c_C));
    return 0;
}

int main(){
    return stub();
}