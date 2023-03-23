#export AMD_COMGR_EMIT_VERBOSE_LOGS=1
#export AMD_COMGR_REDIRECT_LOGS=stderr
#export AMD_COMGR_SAVE_TEMPS=1
#export PATH=/home/yaxunl/git/llvm2/assert/bin:/opt/rocm/bin:$PATH
export PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:$PATH
export AMD_LOG_LEVEL=3
#/opt/rocm/hip/bin/hipcc -ffp-contract=fast-honor-pragmas -fhip-fp32-correctly-rounded-divide-sqrt hiprtc.hip
#/home/yaxunl/git/llvm2/assert/bin/clang++ -v --offload-arch=gfx906 -O3 --hip-link -ffp-contract=fast-honor-pragmas -fhip-fp32-correctly-rounded-divide-sqrt hiprtc.hip
#/home/yaxunl/git/llvm2/assert/bin/clang++ --offload-arch=gfx906 -O3 --hip-link -mamdgpu-ieee -save-temps hiprtc.hip
#/home/yaxunl/git/llvm2/assert/bin/clang++ --offload-arch=gfx906 -O3 --hip-link -v -menable-no-nans -mno-amdgpu-ieee -save-temps saxpy.hip
#/home/yaxunl/git/llvm2/assert/bin/clang++ --offload-arch=gfx906 -O3 --hip-link -v -fassociative-math -save-temps saxpy.hip
#/home/yaxunl/git/llvm2/assert/bin/clang++ --offload-arch=gfx906 -O3 --cuda-device-only --hip-link -fslp-vectorize -mllvm -debug-only=SLP -mllvm -print-after=constmerge -save-temps slp.hip
#/home/yaxunl/git/llvm2/assert/bin/clang++ --offload-arch=gfx906 -fopenmp -O3 --cuda-device-only --hip-link -save-temps omp-target.hip -emit-llvm -S -o -
#/opt/rocm/llvm/bin/clang++ --hip-link --offload-arch=gfx906 -fhip-fp32-correctly-rounded-divide-sqrt sqrt.hip
#/home/yaxunl/git/llvm2/assert/bin/clang++ --offload-arch=gfx906 -O3 --hip-link -ftrapping-math -save-temps trap-math.hip
#/home/yaxunl/git/llvm2/assert/bin/clang++ --offload-arch=gfx906 -O3 --hip-link -fopenmp -save-temps omp-target.hip
#/opt/rocm/llvm/bin/clang++ --offload-arch=gfx906 -O3 --hip-link -fopenmp -save-temps omp-target.hip
#/opt/rocm/llvm/bin/clang++ --offload-arch=gfx906 -O0 --hip-link -save-temps atomic-private.hip --cuda-device-only
#clang++ --rocm-path=/opt/rocm --offload-arch=gfx906 -O0 --hip-link -save-temps -ggdb atomic-private.hip
clang++ --rocm-path=/opt/rocm --offload-arch=gfx906 --hip-link -save-temps atomic-managed.hip
#clang++ --rocm-path=/opt/rocm --offload-arch=gfx906 -O3 -ggdb --hip-link -save-temps atomic-private.hip
#./a.out
./a.out
#llvm-dis atomic-private-hip-amdgcn-amd-amdhsa-gfx906.bc
#llvm-extract -func=_Z6kernelv atomic-private-hip-amdgcn-amd-amdhsa-gfx906.ll -o tmp.ll -S
#llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 tmp.ll -print-after-all

#/home/yaxunl/git/llvm2/assert/bin/llvm-dis math-trap-hip-amdgcn-amd-amdhsa-gfx906.bc -o -

#/opt/rocm/llvm/bin/clang++ -O3 --hip-link --offload-arch=gfx906 -fno-hip-fp32-correctly-rounded-divide-sqrt sqrt.hip -save-temps
#./a.out | tee 1.txt
#/opt/rocm/llvm/bin/clang++ -O3 --hip-link --offload-arch=gfx906 -fhip-fp32-correctly-rounded-divide-sqrt sqrt.hip -save-temps
#./a.out | tee 2.txt
#diff 1.txt 2.txt
