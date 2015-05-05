#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "../common/cuda_util.h"
#include "../common/mem_util.h"
#include "../common/math_util.h"

#define NUM_THREADS 128
#define MAX_NUM_THREADS 512

__device__ static inline
float l2_distance_2(const float* device_tr_points,
                    const float* device_cluster_centers,
                    int num_points, int num_coords, int num_clusters,
                    int point_i, int cluster_i)
{
    float dist_sqr_sum = 0;
    float dist;

    for (int coord_i = 0; coord_i < num_coords; coord_i++)
    {
        dist = device_tr_points[num_points * coord_i + point_i]
                - device_cluster_centers[num_clusters * coord_i + cluster_i];
        dist_sqr_sum += sqr(dist);
    }

    return dist_sqr_sum;
}

__global__ static
void assign_clusters(const float *device_tr_points,
                     int num_points, int num_coords, int num_clusters,
                     const float *device_cluster_centers,
                     int *device_cluster_assignments,
                     unsigned int *device_delta_partial_sums,
                     float* device_loss)
{
    extern __shared__ unsigned char shared_delta_partial_sums_uchar[];

    shared_delta_partial_sums_uchar[threadIdx.x] = 0;

    __syncthreads();

    int point_i = blockDim.x * blockIdx.x + threadIdx.x;

    if (point_i < num_points)
    {
        float cur_dist, min_dist;
        int best_cluster;

        best_cluster = 0;

        min_dist = l2_distance_2(device_tr_points, device_cluster_centers,
                                 num_points, num_coords, num_clusters,
                                 point_i, 0);

        for (int cluster_i = 1; cluster_i < num_clusters; cluster_i++)
        {
            cur_dist = l2_distance_2(device_tr_points, device_cluster_centers,
                                     num_points, num_coords, num_clusters,
                                     point_i, cluster_i);

            if (cur_dist < min_dist)
            {
                min_dist = cur_dist;
                best_cluster = cluster_i;
            }
        }

        device_loss[point_i] = min_dist;

        if (device_cluster_assignments[point_i] != best_cluster)
        {
            shared_delta_partial_sums_uchar[threadIdx.x] = 1;
        }

        device_cluster_assignments[point_i] = best_cluster;

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

__global__ static
void update_new_cluster_centers(const float *device_points,
                                int num_coords, int cluster_i, int point_i,
                                float *device_new_cluster_centers)
{
    int coord_i = blockDim.x * blockIdx.x + threadIdx.x;

    if (coord_i < num_coords)
    {
        device_new_cluster_centers[cluster_i * num_coords + coord_i] \
            += device_points[point_i * num_coords + coord_i];
    }
}

__global__ static
void calc_cluster_centers(float *device_cluster_centers,
                          float *device_new_cluster_centers,
                          int num_coords, int num_clusters,
                          int cluster_i,
                          int new_cluster_size)
{
    int coord_i = blockDim.x * blockIdx.x + threadIdx.x;

    if (coord_i < num_coords)
    {
        if (new_cluster_size)
        {
            device_cluster_centers[coord_i * num_clusters + cluster_i] = \
                device_new_cluster_centers[cluster_i * num_coords + coord_i] / new_cluster_size;
        }

        device_new_cluster_centers[cluster_i * num_coords + coord_i] = 0.0;
    }
}


void kmeans(const float **points,
            int num_points, int num_coords, int num_clusters,
            float **cluster_centers, int* cluster_assignments,
            int *total_iter, float *total_loss, float *delta_percent,
            int max_iter, float threshold, unsigned int seed)
{
    /*
        Indices.
    */
    int cluster_i, point_i, coord_i;

    /*
        Copy points to device,
        assume points are allocated with malloc_2d.
    */
    float *device_tr_points, **tr_points;
    float *device_points;

    malloc_2d(tr_points, num_coords, num_points, float);
    for (point_i = 0; point_i < num_points; point_i++)
    {
        for (coord_i = 0; coord_i < num_coords; coord_i++)
        {
            tr_points[coord_i][point_i] = points[point_i][coord_i];
        }
    }

    checkCudaError(
        cudaMalloc((void **)&device_tr_points, num_coords * num_points * sizeof(float)));
    checkCudaError(
        cudaMemcpy(device_tr_points, tr_points[0],
                    num_coords * num_points * sizeof(float), cudaMemcpyHostToDevice));

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
    const unsigned int num_cc_threads = MAX_NUM_THREADS;
    const unsigned int num_cc_blks = (num_coords + num_cc_threads - 1) / num_cc_threads;

    /*
        Allocate device storages.
    */
    float *device_cluster_centers;
    int *device_cluster_assignments;
    unsigned int *device_delta_partial_sums;
    float *device_loss;

    checkCudaError(
        cudaMalloc((void **)&device_cluster_centers, num_clusters * num_coords * sizeof(float)));

    checkCudaError(
        cudaMalloc((void **)&device_cluster_assignments, num_points * sizeof(int)));
    checkCudaError(
        cudaMemset(device_cluster_assignments, -1, num_points * sizeof(int)));

    checkCudaError(
        cudaMalloc((void **)&device_delta_partial_sums, num_r_threads * sizeof(unsigned int)));

    checkCudaError(
        cudaMalloc((void **)&device_loss, num_points * sizeof(float)));

    /*
        For cluster center calculations.
    */
    int *new_cluster_sizes;
    float *device_new_cluster_centers;

    new_cluster_sizes = (int*)calloc(num_clusters, sizeof(int));
    assert(new_cluster_sizes != NULL);

    checkCudaError(
        cudaMalloc((void **)&device_new_cluster_centers,
                    num_clusters * num_coords * sizeof(float)));
    checkCudaError(
        cudaMemset(device_new_cluster_centers, 0,
                    num_clusters * num_coords * sizeof(float)));

    /*
        Randomly init transposed cluster centers for better coalescing.
    */
    if (seed) { srand(seed); } else { srand(time(NULL)); }

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

    /* Copy cluster centers to device. */
    checkCudaError(
        cudaMemcpy(device_cluster_centers, tr_cluster_centers[0],
                    num_clusters * num_coords * sizeof(float), cudaMemcpyHostToDevice));

    unsigned int delta;
    *total_iter = 0;

    do {
        /* Assign clusters. */
        assign_clusters <<< num_blks, num_threads, smem_size >>>
            ((const float*) device_tr_points, num_points, num_coords, num_clusters,
                (const float*) device_cluster_centers,
                device_cluster_assignments,
                device_delta_partial_sums,
                device_loss);

        checkCudaError(cudaDeviceSynchronize());

        /* Compute delta. */
        reduce_delta_partial_dums <<< 1, num_r_threads, r_smem_size >>>
            (device_delta_partial_sums, num_blks);

        checkCudaError(cudaDeviceSynchronize());

        checkCudaError(
            cudaMemcpy(&delta, device_delta_partial_sums,
                        sizeof(unsigned int), cudaMemcpyDeviceToHost));

        *delta_percent = ((float) delta)/((float) num_points);

        /* Calculate new cluster centers. */
        checkCudaError(
            cudaMemcpy(cluster_assignments, device_cluster_assignments,
                        num_points * sizeof(int), cudaMemcpyDeviceToHost));


        for (point_i = 0; point_i < num_points; point_i++)
        {
            cluster_i = cluster_assignments[point_i];

            new_cluster_sizes[cluster_i]++;

            update_new_cluster_centers<<< num_cc_blks, num_cc_threads >>>
                ((const float*) device_points,
                    num_coords, cluster_i, point_i,
                    device_new_cluster_centers);
        }

        checkCudaError(cudaDeviceSynchronize());

        for (cluster_i = 0; cluster_i < num_clusters; cluster_i++)
        {
            calc_cluster_centers <<< num_cc_blks, num_cc_threads >>>
                (device_cluster_centers, device_new_cluster_centers,
                    num_coords, num_clusters, cluster_i,
                    new_cluster_sizes[cluster_i]);

            new_cluster_sizes[cluster_i] = 0;
        }

        checkCudaError(cudaDeviceSynchronize());

    } while (*delta_percent > threshold && ++(*total_iter) < max_iter);

    /*
        Fill in cluster centers.
    */
    checkCudaError(
        cudaMemcpy(tr_cluster_centers[0], device_cluster_centers,
                    num_clusters * num_coords * sizeof(float), cudaMemcpyDeviceToHost));

    for (coord_i = 0; coord_i < num_coords; coord_i++)
    {
        for (cluster_i = 0; cluster_i < num_clusters; cluster_i++)
        {
            cluster_centers[cluster_i][coord_i] = tr_cluster_centers[coord_i][cluster_i];
        }
    }

    /*
        Sum up loss in CPU, since we are only doing this once.
     */
    float *loss = (float*)malloc(num_points * sizeof(float));
    checkCudaError(
        cudaMemcpy(loss, device_loss, num_points * sizeof(float), cudaMemcpyDeviceToHost));

    *total_loss = 0.0;
    for (point_i = 0; point_i < num_points; point_i++)
    {
        *total_loss += sqrt(loss[point_i]);
    }

    /*
        Free memory.
    */
    checkCudaError(cudaFree(device_tr_points));
    checkCudaError(cudaFree(device_points));
    checkCudaError(cudaFree(device_cluster_centers));
    checkCudaError(cudaFree(device_cluster_assignments));
    checkCudaError(cudaFree(device_delta_partial_sums));
    checkCudaError(cudaFree(device_loss));
    checkCudaError(cudaFree(device_new_cluster_centers));

    free_2d(tr_points);
    free_2d(tr_cluster_centers);
    free(new_cluster_sizes);
    free(loss);
}
