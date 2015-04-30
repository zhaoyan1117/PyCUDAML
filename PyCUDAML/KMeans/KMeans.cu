#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <iomanip>
#include <math.h>
#include <time.h>

#include "KMeans.cuh"

#define NUM_THREADS 16

inline unsigned int calc_num_blks(int value)
{
  return (unsigned int) ((value + NUM_THREADS - 1) / NUM_THREADS);
}

void kmeans(int k, const float **X,
            int n, int d,
            int max_iter, float threshold,
            float **cluster_centers, int* cluster_assignments)
{
  srand(time(NULL));

  /* Copy input to GPU as a flatten array */
  float *device_X;
  if (cudaMalloc((void **) &device_X, n * d * sizeof(float)) != cudaSuccess)
    throw;
  for (int i = 0; i < n; i++)
  {
    if (cudaMemcpy(device_X+i*d, X[i], d * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
      throw;
  }

  /* Init device cluster centers as a flatten array */
  float *device_cluster_centers;
  if (cudaMalloc((void **) &device_cluster_centers, k * d * sizeof(float)) != cudaSuccess)
    throw;
  curandState* device_states;
  cudaMalloc((void **) &device_states, k*sizeof(curandState));
  cu_init_cluster_centers <<< calc_num_blks(k), NUM_THREADS >>> \
    (k, (const float*) device_X, n, d, device_cluster_centers, \
      device_states, unsigned(time(NULL)));
  if (cudaDeviceSynchronize() != cudaSuccess)
    throw;

  /* Init device cluster assignments */
  int *device_cluster_assignments;
  if (cudaMalloc((void **) &device_cluster_assignments, n*sizeof(int)) != cudaSuccess)
    throw;
  int *device_changed_clusters;
  if (cudaMalloc((void **) &device_changed_clusters, n*sizeof(int)) != cudaSuccess)
    throw;
  cu_assign_clusters <<< calc_num_blks(n), NUM_THREADS >>> \
    (k, (const float*) device_X, n, d, \
     device_cluster_assignments, (const float*) device_cluster_centers, \
     device_changed_clusters);
  if (cudaDeviceSynchronize() != cudaSuccess)
    throw;

  /* Init for computing delta */
  int cu_delta;
  int delta_threads = 0;
  int delta_blocks = 0;
  getNumBlocksAndThreads(n, calc_num_blks(n), NUM_THREADS,
                         delta_blocks, delta_threads);

  int *device_delta_reduce_outputs;
  if (cudaMalloc((void **) &device_delta_reduce_outputs,
      delta_threads*sizeof(int)) != cudaSuccess)
    throw;
  int *delta_reduce_outputs = (int*)malloc(delta_threads*sizeof(int));

  cu_delta = total_reduce(n, delta_threads, delta_blocks,
                          NUM_THREADS, calc_num_blks(n),
                          delta_reduce_outputs,
                          device_changed_clusters,
                          device_delta_reduce_outputs);




  init_cluster_centers(k, X, n, d, cluster_centers);

  /* First round. */
  int delta;
  float delta_rate;
  int cur_iter = 0;

  delta = assign_clusters(k, X, n, d, cluster_assignments, (const float**) cluster_centers);
  calc_cluster_centers(k, X, n, d, (const int*) cluster_assignments, cluster_centers);
  delta_rate = ((float)delta)/((float)n);

  while (!is_terminated(cur_iter, max_iter, delta_rate, threshold))
  {
    std::cout << '\r' << '[' << cur_iter << "/" << max_iter << ']';
    std::cout.flush();

    delta = assign_clusters(k, X, n, d,
                            cluster_assignments, (const float**) cluster_centers);
    calc_cluster_centers(k, X, n, d, (const int*) cluster_assignments, cluster_centers);

    delta_rate = ((float)delta)/((float)n);
    cur_iter++;
  }

  /* Clean up. */
  cudaFree(device_X);
  cudaFree(device_states);
  cudaFree(device_cluster_centers);
  cudaFree(device_cluster_assignments);
  cudaFree(device_changed_clusters);
  cudaFree(device_delta_reduce_outputs);
  free(delta_reduce_outputs);
}

__global__ void cu_init_cluster_centers(int k, const float *device_X, int n, int d,
                                        float *device_cluster_centers,
                                        curandState *device_states, unsigned long seed)
{
  int my_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (my_id < k)
  {
    curandState my_state = device_states[my_id];
    curand_init(seed, my_id, 0, &my_state);
    int X_i;
    X_i = (int) (curand_uniform(&my_state) * n);
    memcpy(device_cluster_centers + my_id * d, device_X + X_i * d, sizeof(float) * d);
  }
}

__global__ void cu_assign_clusters(int k, const float *device_X, int n, int d,
                                   int *device_cluster_assignments,
                                   const float *device_cluster_centers,
                                   int *device_changed_clusters)
{
  int my_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (my_id < n)
  {
    float cur_dist, best_dist;
    int best_cluster;

    best_dist = CUDART_INF_F;
    best_cluster = -1;

    for (int k_i = 0; k_i < k; k_i++)
    {
      cur_dist = cu_calc_distances(device_X+my_id*d,
                                   device_cluster_centers+k_i*d,
                                   d);
      if (cur_dist < best_dist)
      {
        best_dist = cur_dist;
        best_cluster = k_i;
      }
    }

    if (device_cluster_assignments[my_id] != best_cluster)
    {
      device_changed_clusters[my_id] = 1;
    }
    else
    {
      device_changed_clusters[my_id] = 0;
    }

    device_cluster_assignments[my_id] = best_cluster;
  }
}

bool is_terminated(int cur_iter, int max_iter, float delta_rate, float threshold)
{
  if (max_iter)
  {
    if (cur_iter > max_iter)
    {
      std::cout << '\r'
                << "Iteration: ["
                << cur_iter-1 << "/" << max_iter
                << "] | Delta rate: "
                << delta_rate
                << std::endl;
      return 1;
    }
    else if (delta_rate < threshold)
    {
      std::cout << '\r'
                << "Iteration: ["
                << cur_iter-1 << "/" << max_iter
                << "] | Delta rate: "
                << delta_rate
                << std::endl;
      return 1;
    }
    else
    {
      return 0;
    }
  }
  else
  {
    if (delta_rate < threshold)
    {
      std::cout << '\r'
                << "Iteration: ["
                << cur_iter-1 << "/" << max_iter
                << "] | Delta rate: "
                << delta_rate
                << std::endl;
      return 1;
    }
    else
    {
      return 0;
    }
  }
}

void init_cluster_centers(int k, const float **X, int n, int d, float **cluster_centers)
{
  for (int k_i = 0; k_i < k; k_i++)
  {
    int X_i = rand() % n;
    if (!(cluster_centers[k_i] = (float*)malloc(d*sizeof(float))))
    {
      throw;
    }
    memcpy(cluster_centers[k_i], X[X_i], d*sizeof(float));
  }
}

int assign_clusters(int k, const float **X, int n, int d,
          int *cluster_assignments, const float **cluster_centers)
{
  float cur_dist, best_dist;
  int best_cluster;
  int delta = 0;

  for (int X_i = 0; X_i < n; X_i++)
  {
    best_dist = INFINITY;
    best_cluster = -1;

    for (int k_i = 0; k_i < k; k_i++)
    {
      cur_dist = calc_distances(X[X_i], cluster_centers[k_i], d);
      if (cur_dist < best_dist)
      {
        best_dist = cur_dist;
        best_cluster = k_i;
      }
    }

    if (cluster_assignments[X_i] != best_cluster)
    {
      delta++;
    }

    cluster_assignments[X_i] = best_cluster;
  }

  return delta;
}

void calc_cluster_centers(int k, const float **X, int n, int d,
                          const int *cluster_assignments, float **cluster_centers)
{
  float **new_cluster_centers = NULL;
  if (!(new_cluster_centers = (float**)malloc(k*sizeof(float*))))
  {
    throw;
  }
  for (int k_i = 0; k_i < k; k_i++)
  {
    if (!(new_cluster_centers[k_i] = (float*)calloc(d,sizeof(float))))
    {
      throw;
    }
  }

  int *counts = (int*)calloc(k,sizeof(int));
  int cluster;
  for (int X_i = 0; X_i < n; X_i++)
  {
    cluster = cluster_assignments[X_i];

    counts[cluster]++;
    increment(new_cluster_centers[cluster], X[X_i], d);
  }

  for (int k_i = 0; k_i < k; k_i++)
  {
    if (counts[k_i])
    {
      divide(new_cluster_centers[k_i], counts[k_i], d);
    }
  }

  for (int k_i = 0; k_i < k; k_i++)
  {
    if (counts[k_i])
    {
      memcpy(cluster_centers[k_i], new_cluster_centers[k_i], d*sizeof(float));
    }
  }

  free_cluster_centers(k, new_cluster_centers, d);
  free(counts);
}

void increment(float* target, const float* value, int d)
{
  for (int d_i = 0; d_i < d; d_i++)
  {
    target[d_i] += value[d_i];
  }
}

void divide(float* target, float value, int d)
{
  for (int d_i = 0; d_i < d; d_i++)
  {
    target[d_i] = target[d_i] / value;
  }
}

float calc_distances(const float* p1, const float* p2, int d)
{
  float dist_sum = 0;
  float dist = 0;

  for (int d_i = 0; d_i < d; d_i++)
  {
    dist = p1[d_i] - p2[d_i];
    dist_sum += dist * dist;
  }

  return sqrt(dist_sum);
}

__device__ float cu_calc_distances(const float* p1, const float* p2, int d)
{
  float dist_sum = 0;
  float dist = 0;

  for (int d_i = 0; d_i < d; d_i++)
  {
    dist = p1[d_i] - p2[d_i];
    dist_sum += dist * dist;
  }

  return sqrtf(dist_sum);
}

void free_cluster_centers(int k, float **cluster_centers, int d)
{
  for (int k_i = 0; k_i < k; k_i++)
  {
    free(cluster_centers[k_i]);
  }
  free(cluster_centers);
}
