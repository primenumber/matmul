#include "matmul_gpu.h"
#include "matmul_common.hpp"
#include <cuda.h>

__global__ void matmul_kernel_ver1(
    const float * const mat1, const float * const mat2, float * const mat3,
    const size_t pitch1, const size_t pitch2, const size_t pitch3,
    const size_t size1, const size_t size2, const size_t size3) {
  for (size_t i = 0; i < size1; ++i) {
    for (size_t j = 0; j < size3; ++j) {
      for (size_t k = 0; k < size2; ++k) {
        mat3[i*pitch3 + j] += mat1[i*pitch1 + k] * mat2[k*pitch2 + j];
      }
    }
  }
}

__global__ void matmul_kernel_ver2(
    const float * const mat1, const float * const mat2, float * const mat3,
    const size_t pitch1, const size_t pitch2, const size_t pitch3,
    const size_t size1, const size_t size2, const size_t size3) {
  const size_t offset_x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t offset_y = blockIdx.y * blockDim.y + threadIdx.y;
  for (size_t i = offset_x; i < size1; i += blockDim.x * gridDim.x) {
    for (size_t j = offset_y; j < size3; j += blockDim.y * gridDim.y) {
      for (size_t k = 0; k < size2; ++k) {
        mat3[i*pitch3 + j] += mat1[i*pitch1 + k] * mat2[k*pitch2 + j];
      }
    }
  }
}

constexpr unsigned int threadsxPerBlock = 32;
constexpr unsigned int threadsyPerBlock = 32;

#define FMA(index, id) tmp##id += __shfl_sync(0xFFFFFFFF, buf_reg, index, threadsxPerBlock) * local[index][threadIdx.x];

__device__ float fma_impl(const float * const mat1, const float * mat2,
    const size_t size1, const size_t pitch1, const size_t pitch2,
    const size_t i, const size_t j, const size_t k) {
  __shared__ float local[threadsyPerBlock][threadsxPerBlock];
  __syncthreads();
  local[threadIdx.y][threadIdx.x] = __ldg(mat2 + (k+threadIdx.y)*pitch2 + j);
  __syncthreads();
  float buf_reg;
  if (i < size1) {
    buf_reg = __ldg(mat1 + i * pitch1 + k + threadIdx.x);
  }
  const float * mat2_sub = mat2 + k * pitch2 + j;
  float tmp0 = 0.0, tmp1 = 0.0;
  FMA( 0, 0)
  FMA( 1, 0)
  FMA( 2, 0)
  FMA( 3, 0)
  FMA( 4, 0)
  FMA( 5, 0)
  FMA( 6, 0)
  FMA( 7, 0)
  FMA( 8, 0)
  FMA( 9, 0)
  FMA(10, 0)
  FMA(11, 0)
  FMA(12, 0)
  FMA(13, 0)
  FMA(14, 0)
  FMA(15, 0)
  FMA(16, 0)
  FMA(17, 0)
  FMA(18, 0)
  FMA(19, 0)
  FMA(20, 0)
  FMA(21, 0)
  FMA(22, 0)
  FMA(23, 0)
  FMA(24, 0)
  FMA(25, 0)
  FMA(26, 0)
  FMA(27, 0)
  FMA(28, 0)
  FMA(29, 0)
  FMA(30, 0)
  FMA(31, 0)
  return tmp0 + tmp1;
}

__global__ void matmul_kernel_ver3(
    const float * const mat1, const float * const mat2, float * const mat3,
    const size_t pitch1, const size_t pitch2, const size_t pitch3,
    const size_t size1, const size_t size2, const size_t size3) {
  size_t i = threadIdx.y + blockIdx.y * threadsyPerBlock;
  size_t j = threadIdx.x + blockIdx.x * threadsxPerBlock;
  float tmp = 0.0;
  float buf_reg = 0.0;
  for (size_t k = 0; k < size2; k += threadsxPerBlock) {
    if (size2 - k < threadsxPerBlock) {
      size_t ek = size2 - k;
      if (i < size1) {
        if (threadIdx.x < ek) {
          buf_reg = __ldg(mat1 + i * pitch1 + k + threadIdx.x);
        }
      }
      for (size_t k2 = 0; k2 < ek; ++k2) {
        tmp += __shfl_sync(0xFFFFFFFF, buf_reg, k2, threadsxPerBlock) * __ldg(mat2 + (k+k2) * pitch2 + j);
      }
    } else {
      tmp += fma_impl(mat1, mat2, size1, pitch1, pitch2, i, j, k);
    }
  }
  if (i < size1 && j < size3) {
    mat3[i * pitch3 + j] = tmp;
  }
}

__host__ void handle_error(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "[Error] %s (error code: %d)\n", cudaGetErrorString(err), err);
  }
}

__host__ void matmul_gpu_ver1(const float* const mat1, const float * const mat2, float * const mat3,
    const size_t size1, const size_t size2, const size_t size3,
    const size_t pitch1, const size_t pitch2) {
  cudaEvent_t start, stop;
  handle_error(cudaEventCreate(&start));
  handle_error(cudaEventCreate(&stop));
  float *mat1_d;
  float *mat2_d;
  float *mat3_d;
  size_t pitch1_d, pitch2_d, pitch3_d;
  handle_error(cudaMallocPitch((void**)&mat1_d, &pitch1_d, size2 * sizeof(float), size1));
  handle_error(cudaMallocPitch((void**)&mat2_d, &pitch2_d, size3 * sizeof(float), size2));
  handle_error(cudaMallocPitch((void**)&mat3_d, &pitch3_d, size3 * sizeof(float), size1));
  handle_error(cudaMemcpy2D((void*)mat1_d, pitch1_d, (void*)mat1, pitch1 * sizeof(float), size2 * sizeof(float), size1, cudaMemcpyHostToDevice));
  handle_error(cudaMemcpy2D((void*)mat2_d, pitch2_d, (void*)mat2, pitch2 * sizeof(float), size3 * sizeof(float), size2, cudaMemcpyHostToDevice));
  handle_error(cudaMemset2D((void*)mat3_d, pitch3_d, 0, size3 * sizeof(float), size1));
  handle_error(cudaEventRecord(start, 0));
  matmul_kernel_ver1<<<1, 1>>>(
      mat1_d, mat2_d, mat3_d,
      pitch1_d/sizeof(float), pitch2_d/sizeof(float), pitch3_d/sizeof(float),
      size1, size2, size3);
  handle_error(cudaEventRecord(stop, 0));
  handle_error(cudaEventSynchronize(stop));
  handle_error(cudaMemcpy2D((void*)mat3, pitch2 * sizeof(float), (void*)mat3_d, pitch3_d, size3 * sizeof(float), size1, cudaMemcpyDeviceToHost));
  cudaFree(mat1_d);
  cudaFree(mat2_d);
  cudaFree(mat3_d);
  size_t ops = 2 * size1 * size2 * size3;
  float elapsed;
  handle_error(cudaEventElapsedTime(&elapsed, start, stop));
  fprintf(stderr, "Time: %fms, %f GFLOPS\n", elapsed, ops / elapsed / 1e6);
}

__host__ void matmul_gpu_ver2(const float* const mat1, const float * const mat2, float * const mat3,
    const size_t size1, const size_t size2, const size_t size3,
    const size_t pitch1, const size_t pitch2) {
  cudaEvent_t start, stop;
  handle_error(cudaEventCreate(&start));
  handle_error(cudaEventCreate(&stop));
  float *mat1_d;
  float *mat2_d;
  float *mat3_d;
  size_t pitch1_d, pitch2_d, pitch3_d;
  handle_error(cudaMallocPitch((void**)&mat1_d, &pitch1_d, size2 * sizeof(float), size1));
  handle_error(cudaMallocPitch((void**)&mat2_d, &pitch2_d, size3 * sizeof(float), size2));
  handle_error(cudaMallocPitch((void**)&mat3_d, &pitch3_d, size3 * sizeof(float), size1));
  handle_error(cudaMemcpy2D((void*)mat1_d, pitch1_d, (void*)mat1, pitch1 * sizeof(float), size2 * sizeof(float), size1, cudaMemcpyHostToDevice));
  handle_error(cudaMemcpy2D((void*)mat2_d, pitch2_d, (void*)mat2, pitch2 * sizeof(float), size3 * sizeof(float), size2, cudaMemcpyHostToDevice));
  handle_error(cudaMemset2D((void*)mat3_d, pitch3_d, 0, size3 * sizeof(float), size1));
  size_t bv = (size1+threadsyPerBlock-1) / threadsyPerBlock;
  size_t bh = (size3+threadsxPerBlock-1) / threadsxPerBlock;
  dim3 block(threadsxPerBlock, threadsyPerBlock);
  dim3 grid(bh, bv);
  handle_error(cudaEventRecord(start, 0));
  matmul_kernel_ver2<<<grid, block>>>(
      mat1_d, mat2_d, mat3_d,
      pitch1_d/sizeof(float), pitch2_d/sizeof(float), pitch3_d/sizeof(float),
      size1, size2, size3);
  handle_error(cudaEventRecord(stop, 0));
  handle_error(cudaEventSynchronize(stop));
  handle_error(cudaMemcpy2D((void*)mat3, pitch2 * sizeof(float), (void*)mat3_d, pitch3_d, size3 * sizeof(float), size1, cudaMemcpyDeviceToHost));
  cudaFree(mat1_d);
  cudaFree(mat2_d);
  cudaFree(mat3_d);
  size_t ops = 2 * size1 * size2 * size3;
  float elapsed;
  handle_error(cudaEventElapsedTime(&elapsed, start, stop));
  fprintf(stderr, "Time: %fms, %f GFLOPS\n", elapsed, ops / elapsed / 1e6);
}

__host__ void matmul_gpu_ver3(const float* const mat1, const float * const mat2, float * const mat3,
    const size_t size1, const size_t size2, const size_t size3,
    const size_t pitch1, const size_t pitch2) {
  cudaEvent_t start, stop;
  handle_error(cudaEventCreate(&start));
  handle_error(cudaEventCreate(&stop));
  float *mat1_d;
  float *mat2_d;
  float *mat3_d;
  size_t pitch1_d, pitch2_d, pitch3_d;
  handle_error(cudaMallocPitch((void**)&mat1_d, &pitch1_d, size2 * sizeof(float), size1));
  handle_error(cudaMallocPitch((void**)&mat2_d, &pitch2_d, size3 * sizeof(float), size2));
  handle_error(cudaMallocPitch((void**)&mat3_d, &pitch3_d, size3 * sizeof(float), size1));
  handle_error(cudaMemcpy2D((void*)mat1_d, pitch1_d, (void*)mat1, pitch1 * sizeof(float), size2 * sizeof(float), size1, cudaMemcpyHostToDevice));
  handle_error(cudaMemcpy2D((void*)mat2_d, pitch2_d, (void*)mat2, pitch2 * sizeof(float), size3 * sizeof(float), size2, cudaMemcpyHostToDevice));
  handle_error(cudaMemset2D((void*)mat3_d, pitch3_d, 0, size3 * sizeof(float), size1));
  size_t bv = (size1+threadsyPerBlock-1) / threadsyPerBlock;
  size_t bh = (size3+threadsxPerBlock-1) / threadsxPerBlock;
  dim3 block(threadsxPerBlock, threadsyPerBlock);
  dim3 grid(bh, bv);
  handle_error(cudaEventRecord(start, 0));
  matmul_kernel_ver3<<<grid, block>>>(
      mat1_d, mat2_d, mat3_d,
      pitch1_d/sizeof(float), pitch2_d/sizeof(float), pitch3_d/sizeof(float),
      size1, size2, size3);
  handle_error(cudaEventRecord(stop, 0));
  handle_error(cudaEventSynchronize(stop));
  handle_error(cudaMemcpy2D((void*)mat3, pitch2 * sizeof(float), (void*)mat3_d, pitch3_d, size3 * sizeof(float), size1, cudaMemcpyDeviceToHost));
  cudaFree(mat1_d);
  cudaFree(mat2_d);
  cudaFree(mat3_d);
  size_t ops = 2 * size1 * size2 * size3;
  float elapsed;
  handle_error(cudaEventElapsedTime(&elapsed, start, stop));
  fprintf(stderr, "Time: %fms, %f GFLOPS\n", elapsed, ops / elapsed / 1e6);
}
