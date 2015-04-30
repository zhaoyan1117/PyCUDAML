#ifndef __CUDA_HELPER_H__
#define __CUDA_HELPER_H__

#include <stdio.h>

#include "cuda.h"

inline void checkCudaError(cudaError_t e) {
    if (e != cudaSuccess) {
        printf("CUDA error %d, %s\n", e, cudaGetErrorString(e));
    }
}

#endif
