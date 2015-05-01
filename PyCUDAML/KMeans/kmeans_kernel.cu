#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "../common/cuda_util.h"
#include "../common/mem_util.h"
#include "../common/math_util.h"

#define NUM_THREADS 128

__device__ static inline
float l2_distance_2(const float* device_points,
                    const float* device_cluster_centers,
                    int num_points, int num_coords, int num_clusters,
                    int point_i, int cluster_i)
{
    float dist_sqr_sum = 0;
    float dist;

    for (int coord_i = 0; coord_i < num_coords; coord_i++)
    {
        dist = device_points[num_points * coord_i + point_i]
                - device_cluster_centers[num_clusters * coord_i + cluster_i];
        dist_sqr_sum += sqr(dist);
    }

    return dist_sqr_sum;
}

__global__ static
void assign_clusters(const float *device_points,
                     int num_points, int num_coords, int num_clusters,
                     const float *device_cluster_centers,
                     int *device_cluster_assignments,
                     unsigned int *device_delta_partial_sums)
{
    extern __shared__ unsigned char shared_delta_partial_sums_uchar[];

    shared_delta_partial_sums_uchar[threadIdx.x] = 0;

    __syncthreads();

    int my_point_i = blockDim.x * blockIdx.x + threadIdx.x;

    if (my_point_i < num_points)
    {
        float cur_dist, min_dist;
        int best_cluster;

        best_cluster = 0;

        min_dist = l2_distance_2(device_points, device_cluster_centers,
                                 num_points, num_coords, num_clusters,
                                 my_point_i, 0);

        for (int cluster_i = 1; cluster_i < num_clusters; cluster_i++)
        {
            cur_dist = l2_distance_2(device_points, device_cluster_centers,
                                     num_points, num_coords, num_clusters,
                                     my_point_i, cluster_i);

            if (cur_dist < min_dist)
            {
                min_dist = cur_dist;
                best_cluster = cluster_i;
            }
        }

        if (device_cluster_assignments[my_point_i] != best_cluster)
        {
            shared_delta_partial_sums_uchar[threadIdx.x] = 1;
        }

        device_cluster_assignments[my_point_i] = best_cluster;

        __syncthreads();

        /* Reduction */
        for (unsigned int delta_i = blockDim.x / 2; delta_i > 0; delta_i /= 2)
        {
            if (threadIdx.x < delta_i)
            {
                shared_delta_partial_sums_uchar[threadIdx.x] \
                    += shared_delta_partial_sums_uchar[threadIdx.x + delta_i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            device_delta_partial_sums[blockIdx.x] = shared_delta_partial_sums_uchar[0];
        }
    }
}

__global__ static
void reduce_delta_partial_dums(unsigned int *device_delta_partial_sums, int num_partial_sums)
{
    extern __shared__ unsigned int shared_delta_partial_sums_uint[];

    shared_delta_partial_sums_uint[threadIdx.x] = \
        (threadIdx.x < num_partial_sums) ? device_delta_partial_sums[threadIdx.x] : 0;

    __syncthreads();

    for (unsigned int delta_i = next_pow_2(num_partial_sums) / 2; delta_i > 0; delta_i /= 2)
    {
        if (threadIdx.x < delta_i)
        {
            shared_delta_partial_sums_uint[threadIdx.x] \
                += shared_delta_partial_sums_uint[threadIdx.x + delta_i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        device_delta_partial_sums[blockIdx.x] = shared_delta_partial_sums_uint[0];
    }
}

void kmeans(const float **points,
            int num_points, int num_coords, int num_clusters,
            float **cluster_centers, int* cluster_assignments, int *total_iter,
            int max_iter, float threshold, unsigned int seed)
{
    /*
        Indices.
    */
    int cluster_i, point_i, coord_i;

    /*
        Copy transposed points to device for better coalescing.
    */
    float *device_points, **tr_points;

    malloc_2d(tr_points, num_coords, num_points, float);
    for (point_i = 0; point_i < num_points; point_i++)
    {
        for (coord_i = 0; coord_i < num_coords; coord_i++)
        {
            tr_points[coord_i][point_i] = points[point_i][coord_i];
        }
    }

    checkCudaError(
        cudaMalloc((void **)&device_points, num_coords * num_points * sizeof(float)));
    checkCudaError(
        cudaMemcpy(device_points, tr_points[0],
                    num_coords * num_points * sizeof(float), cudaMemcpyHostToDevice));

    /*
        Calculate block size and shared memory size.
    */
    const unsigned int num_threads = NUM_THREADS;
    const unsigned int num_blks = (num_points + num_threads - 1) / num_threads;
    const unsigned int smem_size = num_threads * sizeof(unsigned char);
    const unsigned int num_r_threads = next_pow_2(num_blks);
    const unsigned int r_smem_size = num_r_threads * sizeof(unsigned int);

    /*
        Allocate device storages.
    */
    float *device_cluster_centers;
    int *device_cluster_assignments;
    unsigned int *device_delta_partial_sums;

    checkCudaError(
        cudaMalloc((void **)&device_cluster_centers, num_clusters * num_coords * sizeof(float)));

    checkCudaError(
        cudaMalloc((void **)&device_cluster_assignments, num_points * sizeof(int)));
    checkCudaError(
        cudaMemset(device_cluster_assignments, -1, num_points * sizeof(int)));

    checkCudaError(
        cudaMalloc((void **)&device_delta_partial_sums, num_r_threads * sizeof(unsigned int)));

    /*
        For cluster center calculations.
    */
    int *new_cluster_sizes;
    float **new_cluster_centers;

    new_cluster_sizes = (int*)calloc(num_clusters, sizeof(int));
    assert(new_cluster_sizes != NULL);

    malloc_2d(new_cluster_centers, num_clusters, num_coords, float);
    memset(new_cluster_centers[0], 0, num_clusters * num_coords * sizeof(float));

    /*
        Randomly init transposed cluster centers for better coalescing.
    */
    if (seed)
    {
        srand(seed);
    }
    else
    {
        srand(time(NULL));
    }

    float **tr_cluster_centers;
    malloc_2d(tr_cluster_centers, num_coords, num_clusters, float);

    for (cluster_i = 0; cluster_i < num_clusters; cluster_i++)
    {
        point_i = rand() % num_points;
        for (coord_i = 0; coord_i < num_coords; coord_i++)
        {
            tr_cluster_centers[coord_i][cluster_i] = points[point_i][coord_i];
        }
    }

    unsigned int delta;
    *total_iter = 0;

    do {
        /* Copy cluster centers to device. */
        checkCudaError(
            cudaMemcpy(device_cluster_centers, tr_cluster_centers[0],
                        num_clusters * num_coords * sizeof(float), cudaMemcpyHostToDevice));

        /* Assign clusters. */
        assign_clusters <<< num_blks, num_threads, smem_size >>>
            ((const float*) device_points, num_points, num_coords, num_clusters,
                (const float*) device_cluster_centers,
                device_cluster_assignments,
                device_delta_partial_sums);

        checkCudaError(cudaDeviceSynchronize());

        /* Compute delta. */
        reduce_delta_partial_dums <<< 1, num_r_threads, r_smem_size >>>
            (device_delta_partial_sums, num_blks);

        checkCudaError(cudaDeviceSynchronize());

        checkCudaError(
            cudaMemcpy(&delta, device_delta_partial_sums,
                        sizeof(unsigned int), cudaMemcpyDeviceToHost));

        /* Calculate new cluster centers. */
        checkCudaError(
            cudaMemcpy(cluster_assignments, device_cluster_assignments,
                        num_points * sizeof(int), cudaMemcpyDeviceToHost));


        for (point_i = 0; point_i < num_points; point_i++)
        {
            cluster_i = cluster_assignments[point_i];

            new_cluster_sizes[cluster_i]++;

            for (coord_i = 0; coord_i < num_coords; coord_i++)
            {
                new_cluster_centers[cluster_i][coord_i] += points[point_i][coord_i];
            }
        }

        for (cluster_i = 0; cluster_i < num_clusters; cluster_i++)
        {
            for (coord_i = 0; coord_i < num_coords; coord_i++)
            {
                if (new_cluster_sizes[cluster_i])
                {
                    tr_cluster_centers[coord_i][cluster_i] =
                        new_cluster_centers[cluster_i][coord_i] / new_cluster_sizes[cluster_i];
                }
                new_cluster_centers[cluster_i][coord_i] = 0.0;
            }
            new_cluster_sizes[cluster_i] = 0;
        }

    } while (((float) delta)/((float) num_points) > threshold && ++(*total_iter) < max_iter);

    /*
        Fill in cluster centers.
    */
    for (coord_i = 0; coord_i < num_coords; coord_i++)
    {
        for (cluster_i = 0; cluster_i < num_clusters; cluster_i++)
        {
            cluster_centers[cluster_i][coord_i] = tr_cluster_centers[coord_i][cluster_i];
        }
    }

    /*
        Free memory.
    */
    checkCudaError(cudaFree(device_points));
    checkCudaError(cudaFree(device_cluster_centers));
    checkCudaError(cudaFree(device_cluster_assignments));
    checkCudaError(cudaFree(device_delta_partial_sums));

    free_2d(tr_points);
    free_2d(tr_cluster_centers);
    free_2d(new_cluster_centers);
    free(new_cluster_sizes);
}
