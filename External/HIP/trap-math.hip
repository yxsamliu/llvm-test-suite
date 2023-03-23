#include <hip/hip_runtime.h>
#include <cmath>

namespace TestMathTrap {
__device__ bool DevPass = false;

__global__ void kernel(double x, double y) {
  DevPass = x/y > 0;
}

bool test() {
  kernel<<<1, 1>>>(1, 0);
  bool Pass;
  hipMemcpyFromSymbol(&Pass, DevPass, sizeof(DevPass));
  return Pass;
}
}

int main() {
  bool Pass = TestMathTrap::test();

  if (!Pass) {
    printf("FAILED!\n");
    return 1;
  }

  printf("PASSED!\n");
  return 0;
}
