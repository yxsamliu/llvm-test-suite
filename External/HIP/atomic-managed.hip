#include "hip/hip_runtime.h"

__global__

void atomicFetchTestKernel(int *atomic_ptr_dev)  {

  int val = 13;

  *atomic_ptr_dev = 2;

  int previous;

  previous =__hip_atomic_fetch_add(atomic_ptr_dev, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

  printf("Add test - Expected previous: %d, actual previous %d\n", 2, previous); //produces correct result 2 for hipMalloc

  previous =__hip_atomic_fetch_or(atomic_ptr_dev, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

  printf("Or test - Expected previous: %d, actual previous %d\n", 15, previous); //produces correct result 15 for hipMalloc

}

 

int main(int argc, char **argv)

{

   int *atomic_ptr_dev;

   //everything works as expected when hipMalloc is used instead of hipMallocManaged

   //hipMalloc(&atomic_ptr_dev, sizeof(int));  

 

   // this reproduces unexpected output atomic_fetch_or

   hipMallocManaged(&atomic_ptr_dev, sizeof(int));

   atomicFetchTestKernel<<<1,1>>>(atomic_ptr_dev);

   hipDeviceSynchronize();

}
