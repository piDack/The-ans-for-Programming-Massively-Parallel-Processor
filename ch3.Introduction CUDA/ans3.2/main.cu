#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>


__global__ void MatMulVec(float* ans,float* mat,float * vec,int dim) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if( index > dim*dim ) return;
    if( blockIdx.x>dim ) return;
    ans[blockIdx.x]+=mat[index]*vec[blockIdx.x];
}

int main(){
    size_t dim=100;
    size_t size=dim*dim*sizeof(float);
    size_t vec_size=dim*sizeof(float);
    float* mat=(float*)malloc(size);
    float* vec=(float*)malloc(vec_size);
    float* ans=(float*)malloc(vec_size);
    float* c_mat,c_vec,c_ans;
    cudaMalloc((void**)&c_mat, size);
    cudaMalloc((void**)&c_mat, vec_size);
    cudaMalloc((void**)&c_mat, vec_size);
    MatMulVec<<<ceil(dim / 256),256>>>(ans,mat,ans,dim);
    cudaFree(mat);
    cudaFree(vec);
    cudaFree(ans);
    return 0;
}