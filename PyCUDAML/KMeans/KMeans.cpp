#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "KMeans.hpp"

void kmeans(int k, const float **X,
            int n, int d,
            int max_iter, float threshold,
            float **cluster_centers, int* cluster_assignments)
{
  srand(time(NULL));

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

void free_cluster_centers(int k, float **cluster_centers, int d)
{
  for (int k_i = 0; k_i < k; k_i++)
  {
    free(cluster_centers[k_i]);
  }
  free(cluster_centers);
}
