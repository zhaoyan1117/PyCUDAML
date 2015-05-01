#ifndef __KMEANS_KERNEL_H__
#define __KMEANS_KERNEL_H__

void kmeans(const float **points,
            int num_points, int num_coords, int num_clusters,
            float **cluster_centers, int* cluster_assignments, int *total_iter,
            int max_iter, float threshold);

#endif /* __KMEANS_KERNEL_H__ */
