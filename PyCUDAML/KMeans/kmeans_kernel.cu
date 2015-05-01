#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "../common/cuda_util.h"
#include "../common/mem_util.h"

#define NUM_THREADS 128

__host__ __device__ static inline
unsigned int next_pow_2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

__host__ __device__ static inline
float l2_distance_2(const float* p1, const float* p2, int len)
{
    float dist_sqr_sum = 0;

    for (int i = 0; i < len; i++)
        dist_sqr_sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);

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
        min_dist = l2_distance_2(device_points + my_point_i * num_coords,
                                 device_cluster_centers,
                                 num_coords);

        for (int cluster_i = 1; cluster_i < num_clusters; cluster_i++)
        {
            cur_dist = l2_distance_2(device_points + my_point_i * num_coords,
                                     device_cluster_centers + cluster_i * num_coords,
                                     num_coords);

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

    for (unsigned int delta_i = next_pow_2(num_partial_sums) / 2; delta_i > 0; delta_i /= 1)
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
            int max_iter, float threshold)
{
    /*
        Indices.
    */
    int cluster_i, point_i, coord_i;

    /*
        Copy points to device,
        assume points are allocated with malloc_2d.
    */
    float *device_points;

    // printf("haha1\n");

    checkCudaError(
        cudaMalloc((void **)&device_points, num_points * num_coords * sizeof(float)));
    checkCudaError(
        cudaMemcpy(device_points, points[0],
                    num_points * num_coords * sizeof(float), cudaMemcpyHostToDevice));

    /*
        Calculate block size and shared memory size.
    */
    const unsigned int num_threads = NUM_THREADS;
    const unsigned int num_blks = (num_points + num_threads - 1) / num_threads;
    const unsigned int smem_size = num_threads * sizeof(unsigned char);
    const unsigned int num_r_threads = next_pow_2(num_blks);
    const unsigned int r_smem_size = num_r_threads * sizeof(unsigned int);

    // printf("haha2\n");

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

    // printf("haha3\n");


    /*
        For cluster center calculations.
    */
    int *new_cluster_sizes;
    float **new_cluster_centers;

    new_cluster_sizes = (int*)calloc(num_clusters, sizeof(int));
    assert(new_cluster_sizes != NULL);

    malloc_2d(new_cluster_centers, num_clusters, num_coords, float);
    memset(new_cluster_centers[0], 0, num_clusters * num_coords * sizeof(float));

    // printf("haha4\n");

    /*
        Randomly init cluster centers,
        assume cluster_centers is allocated with malloc_2d.
    */
    srand(time(NULL));

    for (cluster_i = 0; cluster_i < num_clusters; cluster_i++)
    {
        point_i = rand() % num_points;
        memcpy(cluster_centers[cluster_i], points[point_i],
                num_coords * sizeof(float));
    }

    unsigned int delta;
    *total_iter = 0;

    do {
        /* Copy cluster centers to device. */
        checkCudaError(
            cudaMemcpy(device_cluster_centers, cluster_centers[0],
                        num_clusters * num_coords * sizeof(float), cudaMemcpyHostToDevice));

        // printf("haha5\n");

        /* Assign clusters. */
        assign_clusters <<< num_blks, num_threads, smem_size >>>
            ((const float*) device_points, num_points, num_coords, num_clusters,
                (const float*) device_cluster_centers,
                device_cluster_assignments,
                device_delta_partial_sums);

        // printf("haha6\n");


        checkCudaError(cudaDeviceSynchronize());

        // printf("haha7\n");

        /* Compute delta. */
        reduce_delta_partial_dums <<< 1, num_r_threads, r_smem_size >>>
            (device_delta_partial_sums, num_blks);

        // printf("haha8\n");

        checkCudaError(cudaDeviceSynchronize());


        // printf("haha9\n");

        checkCudaError(
            cudaMemcpy(&delta, device_delta_partial_sums,
                        sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // printf("haha10\n");

        /* Calculate new cluster centers. */
        checkCudaError(
            cudaMemcpy(cluster_assignments, device_cluster_assignments,
                        num_points * sizeof(int), cudaMemcpyDeviceToHost));

        // printf("haha11\n");

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
                    cluster_centers[cluster_i][coord_i] =
                        new_cluster_centers[cluster_i][coord_i] / new_cluster_sizes[cluster_i];
                }
                new_cluster_centers[cluster_i][coord_i] = 0.0;
            }
            new_cluster_sizes[cluster_i] = 0;
        }

    } while (((float) delta)/((float) num_points) > threshold && ++(*total_iter) < max_iter);

    /*
        Free memory.
    */
    checkCudaError(cudaFree(device_points));
    checkCudaError(cudaFree(device_cluster_centers));
    checkCudaError(cudaFree(device_cluster_assignments));
    checkCudaError(cudaFree(device_delta_partial_sums));

    free_2d(new_cluster_centers);
    free(new_cluster_sizes);
}
