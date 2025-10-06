#include <hip/hip_runtime.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cstdlib>

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            std::cerr << "HIP error: " << hipGetErrorString(err)               \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

struct Results {
  float prev1;
  float prev2;
  float finalVal;
};

// Single-threaded kernels that exercise __scoped_atomic_fetch_min on float
// with two sequential updates. This verifies correct return value and update.
__global__ void test_min_system(float *ptr, Results *out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float p1 = __scoped_atomic_fetch_min(ptr, 50.0f, __ATOMIC_RELAXED,
                                         __MEMORY_SCOPE_SYSTEM);
    float p2 = __scoped_atomic_fetch_min(ptr, 75.0f, __ATOMIC_RELAXED,
                                         __MEMORY_SCOPE_SYSTEM);
    out->prev1 = p1;
    out->prev2 = p2;
    out->finalVal = *ptr;
  }
}

__global__ void test_min_device(float *ptr, Results *out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float p1 = __scoped_atomic_fetch_min(ptr, 50.0f, __ATOMIC_RELAXED,
                                         __MEMORY_SCOPE_DEVICE);
    float p2 = __scoped_atomic_fetch_min(ptr, 75.0f, __ATOMIC_RELAXED,
                                         __MEMORY_SCOPE_DEVICE);
    out->prev1 = p1;
    out->prev2 = p2;
    out->finalVal = *ptr;
  }
}

__global__ void test_min_workgroup(float *ptr, Results *out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float p1 = __scoped_atomic_fetch_min(ptr, 50.0f, __ATOMIC_RELAXED,
                                         __MEMORY_SCOPE_WRKGRP);
    float p2 = __scoped_atomic_fetch_min(ptr, 75.0f, __ATOMIC_RELAXED,
                                         __MEMORY_SCOPE_WRKGRP);
    out->prev1 = p1;
    out->prev2 = p2;
    out->finalVal = *ptr;
  }
}

__global__ void test_min_wavefront(float *ptr, Results *out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float p1 = __scoped_atomic_fetch_min(ptr, 50.0f, __ATOMIC_RELAXED,
                                         __MEMORY_SCOPE_WVFRNT);
    float p2 = __scoped_atomic_fetch_min(ptr, 75.0f, __ATOMIC_RELAXED,
                                         __MEMORY_SCOPE_WVFRNT);
    out->prev1 = p1;
    out->prev2 = p2;
    out->finalVal = *ptr;
  }
}

__global__ void test_min_singlethread(float *ptr, Results *out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float p1 = __scoped_atomic_fetch_min(ptr, 50.0f, __ATOMIC_RELAXED,
                                         __MEMORY_SCOPE_SINGLE);
    float p2 = __scoped_atomic_fetch_min(ptr, 75.0f, __ATOMIC_RELAXED,
                                         __MEMORY_SCOPE_SINGLE);
    out->prev1 = p1;
    out->prev2 = p2;
    out->finalVal = *ptr;
  }
}

static bool approx_equal(float a, float b, float eps = 1e-6f) {
  return std::fabs(a - b) <= eps * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

template <typename Launcher>
static void run_one(const char *label, Launcher launch,
                    float init, float expected_prev1, float expected_prev2,
                    float expected_final) {
  float *dX = nullptr;
  Results *dRes = nullptr;
  Results hRes{};

  HIP_CHECK(hipMalloc(&dX, sizeof(float)));
  HIP_CHECK(hipMalloc(&dRes, sizeof(Results)));

  HIP_CHECK(hipMemcpy(dX, &init, sizeof(float), hipMemcpyHostToDevice));

  launch(dX, dRes);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(&hRes, dRes, sizeof(Results), hipMemcpyDeviceToHost));

  bool ok = approx_equal(hRes.prev1, expected_prev1) &&
            approx_equal(hRes.prev2, expected_prev2) &&
            approx_equal(hRes.finalVal, expected_final);

  if (ok) {
    std::cout << "[PASS] " << label << std::endl;
  } else {
    std::cerr << "[FAIL] " << label
              << " prev1=" << hRes.prev1 << " prev2=" << hRes.prev2
              << " final=" << hRes.finalVal << std::endl;
    std::exit(EXIT_FAILURE);
  }

  HIP_CHECK(hipFree(dRes));
  HIP_CHECK(hipFree(dX));
}

int main() {
  const float init = 100.0f;
  run_one("SYSTEM",
          [](float *dx, Results *dr) { test_min_system<<<1, 1>>>(dx, dr); },
          init, /*prev1*/ 100.0f, /*prev2*/ 50.0f, /*final*/ 50.0f);
  run_one("DEVICE",
          [](float *dx, Results *dr) { test_min_device<<<1, 1>>>(dx, dr); },
          init, /*prev1*/ 100.0f, /*prev2*/ 50.0f, /*final*/ 50.0f);
  run_one("WRKGRP",
          [](float *dx, Results *dr) { test_min_workgroup<<<1, 1>>>(dx, dr); },
          init, /*prev1*/ 100.0f, /*prev2*/ 50.0f, /*final*/ 50.0f);
  run_one("WVFRNT",
          [](float *dx, Results *dr) { test_min_wavefront<<<1, 1>>>(dx, dr); },
          init, /*prev1*/ 100.0f, /*prev2*/ 50.0f, /*final*/ 50.0f);
  run_one("SINGLE",
          [](float *dx, Results *dr) { test_min_singlethread<<<1, 1>>>(dx, dr); },
          init, /*prev1*/ 100.0f, /*prev2*/ 50.0f, /*final*/ 50.0f);

  return 0;
}
