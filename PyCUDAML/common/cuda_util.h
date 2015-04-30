#ifndef __CUDA_UTIL_H__
#define __CUDA_UTIL_H__

#include <stdio.h>

#include "cuda.h"

inline void checkCudaError(cudaError_t e) {
    if (e != cudaSuccess) {
        printf("CUDA Error: %d, %s\n", e, cudaGetErrorString(e));
        exit(1);
    }
}

#endif /* __CUDA_UTIL_H__ */
