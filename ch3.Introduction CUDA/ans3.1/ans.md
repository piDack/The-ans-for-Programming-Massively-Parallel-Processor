## ans 3.1
1. [a kernal](./main.cu)
1. [b kernal](./main.cu)
1. [c kernal](./main.cu)
1. 上述 Kernel 中，使用 a 方法的 Kernel 最佳:
    * 充分的利用了GPU的核心
    * 避免了单个core运行相对复杂的逻辑
