#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include "../../comm/helper_cuda.h"

__global__ void reduce(float* d_in,float *d_out) {
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int t=threadIdx.x;
    paartialSum[t]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();
    for(unsigned int stride=blockDim.x/2;stride>1;stride/=2){
        if(t<stride)
            partialSum[t]+=partialSum[t+stride];
        __syncthreads();
    }
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}

template<typename T>
void rand_array(T * array,size_t len){
    for(int i=0;i<len;++i){
        array[i]=((T)rand())/RAND_MAX;
    }
}
bool check(float* mat,float* res,int dim)
{
    for(int i=0;i<dim;++i){
        float p=0.0;
        for(int j=0;j<dim;++j)
            p+=mat[j];
        if(fabs(p-res[i])>1e-10) return false;
    }
    return true;
}

int main(){
    const size_t dim=300;
    size_t size=dim*dim*sizeof(float);
    float * mat=(float*)malloc(size);
    float * res=(float*)malloc(size/dim);
    if(mat==nullptr || res==nullptr){
        return -1;
    }
    rand_array<float>(mat,dim*dim);
    float * c_mat;
    float * c_res;
    checkCudaErrors(cudaMalloc((void**)&c_mat, size));
    checkCudaErrors(cudaMalloc((void**)&c_res, size/dim));
    checkCudaErrors(cudaMemcpy(c_mat, mat, size, cudaMemcpyHostToDevice));
    for(int i=0;i<dim;++i){
        reduce<<<ceil(dim/512),512>>>(c_mat+dim*i,c_res+i);
    }
    checkCudaErrors(cudaMemcpy(res, c_res, size/dim, cudaMemcpyDeviceToHost));
    if(!check(mat,res))
        std::cout<<"fail\n";
    checkCudaErrors(cudaFree(c_mat));
    checkCudaErrors(cudaFree(c_res));
    free(res);
    free(mat);
}