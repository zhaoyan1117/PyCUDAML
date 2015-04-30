/*
Adaped from cuda 7 sample reduction code.
*/

#ifndef __REDUCTION_H__
#define __REDUCTION_H__

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads,
                            int &blocks, int &threads);

int total_reduce(int n,
                 int numThreads, int numBlocks,
                 int maxThreads, int maxBlocks,
                 int *h_odata, int *d_idata, int *d_odata);

#endif
