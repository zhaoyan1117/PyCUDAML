#include <stdio.h>
#include "KMeans.hpp"

float kmeans(int k, float *X, int N) {
    printf("k is %d\n", k);
    printf("N is %d\n", N);

    float sum = 0;

    for (int n = 0; n < N; n++) {
        sum += X[n];
    }

    printf("Sum is %f\n", sum);

    return sum;
}
