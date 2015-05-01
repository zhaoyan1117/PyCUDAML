#ifndef __KMEANS_KERNEL_H__
#define __KMEANS_KERNEL_H__

void kmeans(const float **points,
            int num_points, int num_coords, int num_clusters,
            float **cluster_centers, int* cluster_assignments,
            int *total_iter, float *total_loss, float *delta_percent,
            int max_iter, float threshold, unsigned int seed);

#endif /* __KMEANS_KERNEL_H__ */
