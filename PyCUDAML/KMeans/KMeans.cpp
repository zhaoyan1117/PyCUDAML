#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "KMeans.hpp"

float kmeans(int k, const float **X,
             int n, int d,
             int max_iter, int threshold,
             float **cluster_centers)
{
    srand(time(NULL));
    init_cluster_centers(k, X, n, d, cluster_centers);
    return 0;
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

void free_cluster_centers(int k, float **cluster_centers, int d)
{
    for (int k_i = 0; k_i < k; k_i++)
    {
        free(cluster_centers[k_i]);
    }
    free(cluster_centers);
}
