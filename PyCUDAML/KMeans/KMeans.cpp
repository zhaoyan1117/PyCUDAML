#include <stdio.h>
#include "KMeans.hpp"

float kmeans(int k, float **X, int n, int d, int max_iter, int threshold) {
    printf("k is %d\n", k);
    printf("n is %d\n", n);
    printf("d is %d\n", d);
    printf("max_iter is %d\n", max_iter);
    printf("threshold is %d\n", threshold);

    float sum = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            sum += X[i][j];
        }
    }

    printf("Sum is %f\n", sum);

    return sum;
}
