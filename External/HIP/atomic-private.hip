#include "hip/hip_runtime.h"

__device__ int aaa;
#define T unsigned long
__global__ void kernel(){
    //char buffer[sizeof(T)];
    T buffer;
    T* t = reinterpret_cast<T*> (&buffer);
    *t = 9;
    __hip_atomic_fetch_min(t, 4, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

int main(){
    kernel<<<1, 1>>>();
    hipDeviceSynchronize();
    return 0;
}

