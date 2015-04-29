float kmeans(int k, const float **X,
             int n, int d,
             int max_iter, int threshold,
             float **cluster_centers);

void init_cluster_centers(int k, const float **X, int n, int d, float **cluster_centers);

void free_cluster_centers(int k, float **cluster_centers, int d);
