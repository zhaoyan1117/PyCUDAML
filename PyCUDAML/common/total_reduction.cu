/*
    Adapted from cuda 7 sample reduction code.
*/

#include "reduction.h"
#include "total_reduction.h"

#include "cuda_helper.h"

#define CPUFINALTHRESHOLD 1

#define WHICHKERNEL 6

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads,
                            int &blocks, int &threads)
{
    //get device capability, to avoid block/grid size excceed the upbound
    cudaDeviceProp prop;
    int device;
    checkCudaError(cudaGetDevice(&device));
    checkCudaError(cudaGetDeviceProperties(&prop, device));

    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);

    if (blocks > prop.maxGridSize[0])
    {
        blocks /= 2;
        threads *= 2;
    }

    blocks = MIN(maxBlocks, blocks);
}

int total_reduce(int n,
                 int numThreads, int numBlocks,
                 int maxThreads, int maxBlocks,
                 int *h_odata, int *d_idata, int *d_odata)
{
    int total = 0;
    bool needReadBack = true;
    int whichKernel = WHICHKERNEL;

    checkCudaError(cudaDeviceSynchronize());

    // execute the kernel
    reduce(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

    // sum partial block sums on GPU
    int s = numBlocks;
    int kernel = whichKernel;

    while (s > CPUFINALTHRESHOLD)
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);
        reduce(s, threads, blocks, kernel, d_odata, d_odata);
        s = (s + (threads*2-1)) / (threads*2);
    }

    if (s > 1)
    {
        // copy result from device to host
        checkCudaError(cudaMemcpy(h_odata, d_odata,
                                  s * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i=0; i < s; i++)
        {
            total += h_odata[i];
        }

        needReadBack = false;
    }

    checkCudaError(cudaDeviceSynchronize());

    if (needReadBack)
    {
        // copy final sum from device to host
        checkCudaError(cudaMemcpy(&total, d_odata, sizeof(int), cudaMemcpyDeviceToHost));
    }

    return total;
}
