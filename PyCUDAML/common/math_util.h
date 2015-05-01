#ifndef __MATH_UTIL_H_
#define __MATH_UTIL_H_

#define sqr(x) ((x)*(x))

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
bool is_pow_2(unsigned int x)
{
    return ((x&(x-1))==0);
}

#endif /* __MATH_UTIL_H_ */
