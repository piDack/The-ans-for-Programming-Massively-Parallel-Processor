#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include "../../comm/helper_cuda.h"
__global__ void MatMulVec(float* ans,float* mat,float * vec,int dim) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if( index > dim*dim ) return;
    if( blockIdx.x>dim ) return;
    ans[blockIdx.x]+=mat[index]*vec[blockIdx.x];
}
template<typename T>
void rand_array(T * array,size_t len){
    for(int i=0;i<len;++i){
        array[i]=((T)rand())/RAND_MAX;
    }
}
void check(float* mat,float* vec,float* ans,size_t dim){
    float mans[100]={0.0};
    for(int i=0;i<dim;++i){
        for(int j=0;j<dim;++j){
            mans[j]+=mat[i*dim+j]*vec[j];
        }
    }
    for(int i=0;i<dim;++i){
        if(fabs(mans[i]-ans[i])>1e-10){
            printf("err");
        }
    }
}

int main(){
    const size_t dim=100;
    size_t size=dim*dim*sizeof(float);
    size_t vec_size=dim*sizeof(float);
    float* mat=(float*)malloc(size);
    float* vec=(float*)malloc(vec_size);
    float* ans=(float*)malloc(vec_size);
    if(mat==nullptr || vec==nullptr || ans==nullptr)
        return -1;
    rand_array<float>(mat,dim*dim);
    rand_array<float>(vec,dim);
    float* c_mat,*c_vec,*c_ans;
    checkCudaErrors(cudaMalloc((void**)&c_mat, size));
    checkCudaErrors(cudaMalloc((void**)&c_vec, vec_size));
    checkCudaErrors(cudaMalloc((void**)&c_ans, vec_size));
    checkCudaErrors(cudaMemcpy(c_mat, mat, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(c_vec, vec, vec_size, cudaMemcpyHostToDevice));
    MatMulVec<<<ceil(dim / 256),256>>>(c_ans,c_mat,c_vec,dim);
    checkCudaErrors(cudaMemcpy(ans, c_ans, vec_size, cudaMemcpyDeviceToHost));
    check(c_mat,c_vec,c_ans,dim);
    checkCudaErrors(cudaFree(mat));
    checkCudaErrors(cudaFree(vec));
    checkCudaErrors(cudaFree(ans));
    return 0;
}