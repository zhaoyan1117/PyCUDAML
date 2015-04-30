#ifndef __MEM_UTIL_H_
#define __MEM_UTIL_H_

#include <assert.h>

/*
    Malloc 2d array backed by a 1d array.
    var[primary_dim][secondary_dim]
*/
#define malloc_2d(var, primary_dim, secondary_dim, type) do {            \
    var = (type**)malloc(primary_dim * sizeof(type*));                   \
    assert(var != NULL);                                                 \
    var[0] = (type*)malloc(primary_dim * secondary_dim * sizeof(type));  \
    assert(var[0] != NULL);                                              \
    for (size_t i = 1; i < primary_dim; i++)                             \
        var[i] = var[i-1] + secondary_dim;                               \
} while (0)

/*
    Free 2d array malloced by malloc_2d.
*/
#define free_2d(var) do {  \
    free(var[0]);          \
    free(var);             \
} while (0)

#endif /* __MEM_UTIL_H_ */
