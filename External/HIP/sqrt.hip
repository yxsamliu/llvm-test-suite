#include <iostream>
#include <iomanip>

#include <hip/hip_runtime.h>

#define N  (0x7F800000)

__global__ void saxpy(float a, float* x, float* y) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) y[tid] = sqrt(x[tid]);
}

int main() {

  const float a = 100.0f;
  float* x = (float*)malloc(N * sizeof(float));
  float* y = (float*)malloc(N * sizeof(float));
  union {
    float f;
    unsigned int i;
  } u;
  // Initialize the input data.
  for (size_t i = 0; i < N; ++i) {
    u.i = i;
    x[i] = u.f;
    y[i] = static_cast<float>(i * 2);
  }

  // Make a copy for the GPU implementation.
  float* d_x;
  float* d_y;
  hipMalloc((void**)&d_x, N * sizeof(float));
  hipMalloc((void**)&d_y, N * sizeof(float));
  hipMemcpy(d_x, x, N * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_y, y, N * sizeof(float), hipMemcpyHostToDevice);

  // CPU implementation of saxpy.
  for (int i = 0; i < N; i++) {
    y[i] = sqrt(x[i]);
  }

  // Launch a GPU kernel to compute the saxpy.
  saxpy<<<(N+255)/256, 256>>>(a, d_x, d_y);

  // Copy the device results to host.
  float* h_y = (float*)malloc(N * sizeof(float));
  hipDeviceSynchronize();
  hipMemcpy(h_y, d_y, N * sizeof(float), hipMemcpyDeviceToHost);

  // Verify the results match CPU.
  int errors = 0;
  for (int i = 0; i < N && errors < 10; i++) {
    float err = fabs(y[i] - h_y[i]);
    if (err > fabs(y[i] * 0.0001f)) {
      std::cout << std::setprecision(10) << x[i] << " " << y[i] << " " << h_y[i] << std::endl;
      errors++;
    }
  }
  if (errors != 0)
    std::cout << errors << " errors" << std::endl;
  else
    std::cout << "PASSED!" << std::endl;

  free(h_y);
  free(x);
  free(y);
  hipFree(d_x);
  hipFree(d_y);
  return errors;
}
