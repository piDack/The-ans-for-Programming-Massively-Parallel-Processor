1. 使用malloc是在内存中分配了空间，而cudaMalloc是在显存中分配空间，两者的物理空间是不一样的。
2. 这样分开声明具有很好的灵活性。
3. CUDA中也存在即可在CPU中访问，也可在GPU访问的方式。例如:Unified Memory 。