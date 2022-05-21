## 8.1

P[0]=0\*M[0]+0\*M[1]+N[0]\*M[2]+N[1]\*M[3]+N[2]\*M[4]=(0+0+5+8+9)=22

## 8.2

```
P[0]=M[0]*0 + M[1]*N[0] + M[2]*N[1]  = 8
P[1]=M[0]*N[0] + M[1]*N[1]+M[2]*N[2] = 25
P[2]=M[0]*N[1] + M[1]*N[2]+M[2]*N[3] = 14
P[3]=M[0]*N[2] + M[1]*N[3]+M[2]*N[4] = 22
P[4]=M[0]*N[3] + M[1]*N[4]+M[2]*0    = 7
```

## 8.3

a. 不做处理
b. 右移一位
c. 左移一位
d. 求导
e. 加权平均 

## 8.4

a. m-1  b. m\*n  c. m\*n - m-1 

## 8.5
1. (n+m-1)\*(m-1) + (n\*(m-1)) = (m-1)\*(2n + m - 1)
2. $n^2 * m^2$
3. $(n2 * m2) - (m-1)*(2n + m - 1)$


## 8.6
1. (n2 + m2-1)*(m1-1) + (n1(m2-1))
2. (n1\*n2)*(m1\*m2)
3. (n1\*n2)\*(m1\*m2) - (n2 + m2-1)\*(m1-1) + (n1(m2-1))

## 8.7
1. ${\lceil {n\over t} \rceil }$
2. t
3. ${\lceil {n\over t} \rceil } * (t+m-1)$
4. pic 8_13
    1. ${\lceil {n\over t} \rceil }$
    2. t
    3. ${\lceil {n\over t} \rceil } * t$

## 8.8

```c++
__global__void convolution_2D_basic_kernel(float *N, float *P, int height, int width){ 
    int Row = blockIdx.y * blockDim.y + threadIdx.y; 
    int Col = blockIdx.x * blockDim.x + threadIdx.x; 
    if ( Row < height && Col < width) { 
        int y = Row - (MASK_WIDTH / 2); 
        int x = Col - (MASK_WIDTH / 2); 
        float Pvalue = 0.0f; 
        
        for (int i = 0; i < MASK_WIDTH; i++) { 
            if (y + i >= 0 && y + i < height) { 
                for (int j = 0; j < MASK_WIDTH; j++) { 
                    if (x + j >= 0 && x + j < width) 
                        Pvalue += N[(y+i) * width + (x+j)] * M[i][j]; 
                    } 
            } 
        } 
        P[ Row * width + Col] = Pvalue; 
    } 
}
```

## 8.9

```c++
__global__ 
void convolution_2D_basic_kernel(float *N, float *P, int height, int width, const int Mask_Width) { 
    int tx = threadIdx.x, ty = threadIdx.y; 
    int row_o = blockIdx.y * O_TILE_WIDTH + ty; 
    int col_o = blockIdx.x * O_TILE_WIDTH + tx; 
    int row_i = row_o - (Mask_Width / 2); 
    int col_i = col_o - (Mask_Width / 2); 
    __shared__ float N_ds[TILE_SIZE+MAX_MASK_WIDTH-1][TILE_SIZE+MAX_MASK_WIDTH-1]; 
    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)) 
        N_ds[ty][tx] = N[row_i * width + col_i]; 
    else
        N_ds[ty][tx] = 0.0f; 
        __syncthreads(); 
    float Pvalue = 0.0f; 
    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) { 
        for (int i = 0; i < Mask_Width; i++) { 
            for (int j = 0; j < Mask_Width; j++) 
            Pvalue += N_ds[i+ty][j+tx] * M[i * Mask_Width + j]; 
        } 
        if (row_o < height && col_o < width) 
            P[row_o * width + col_o] = Pvalue; 
    } 
}

```


## 8.10

```c++
__global__ 
void convolution_2D_basic_kernel(float *N, float *P, int height, int width, const int
Mask_Width) { 
    int tx = threadIdx.x, ty = threadIdx.y; 
    int row_o = blockIdx.y * O_TILE_WIDTH + ty; 
    int col_o = blockIdx.x * O_TILE_WIDTH + tx; 
    int row_i = row_o - (Mask_Width / 2); 
    int col_i = col_o - (Mask_Width / 2); 
    __shared__ float N_ds[TILE_SIZE+MAX_MASK_WIDTH-1][TILE_SIZE+MAX_MASK_WIDTH-1]; 
    if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)) 
        N_ds[ty][tx] = N[row_i * width + col_i]; 
    else
        N_ds[ty][tx] = 0.0f; 
    __syncthreads(); 
    float Pvalue = 0.0f; 
    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) { 
        for (int i = 0; i < Mask_Width; i++) { 
            for (int j = 0; j < Mask_Width; j++) 
                Pvalue += N_ds[i+ty][j+tx] * M[i * Mask_Width + j]; 
            } 
        if (row_o < height && col_o < width) 
            P[row_o * width + col_o] = Pvalue; 
    } 
}

```