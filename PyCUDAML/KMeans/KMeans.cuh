#ifndef __KMEANS_CUH__
#define __KMEANS_CUH__

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math_functions.hpp>
#include <math_constants.h>

void kmeans(int k, const float **X,
            int n, int d,
            int max_iter, float threshold,
            float **cluster_centers, int* cluster_assignments);

void init_cluster_centers(int k, const float **X, int n, int d, float **cluster_centers);

float calc_distances(const float* p1, const float* p2, int d);

void increment(float* target, const float* value, int d);

void divide(float* target, float value, int d);

void free_cluster_centers(int k, float **cluster_centers, int d);

int assign_clusters(int k, const float **X, int n, int d,
                    int *cluster_assignments, const float **cluster_centers);

void calc_cluster_centers(int k, const float **X, int n, int d,
                          const int *cluster_assignments, float **cluster_centers);

bool is_terminated(int cur_iter, int max_iter, float delta_rate, float threshold);

__global__ void cu_init_cluster_centers(int k, const float *device_X, int n, int d,
                                        float *device_cluster_centers,
                                        curandState *device_states, unsigned long seed);

__global__ void cu_assign_clusters(int k, const float *device_X, int n, int d,
                                   int *device_cluster_assignments,
                                   const float *device_cluster_centers,
                                   int *device_changed_clusters);

__device__ float cu_calc_distances(const float* p1, const float* p2, int d);

#endif /* __KMEANS_CUH__ */
