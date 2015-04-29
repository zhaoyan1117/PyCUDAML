#ifndef __KMEANS_HPP__
#define __KMEANS_HPP__

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

#endif /* __KMEANS_HPP__ */
