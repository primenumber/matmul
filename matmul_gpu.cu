#include "matmul_gpu.h"
#include <cstdio>

constexpr unsigned int threadsPerBlock = 8;

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
  const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size1 && j < size3) {
    for (size_t k = 0; k < size2; ++k) {
      mat3[i*pitch3 + j] += mat1[i*pitch1 + k] * mat2[k*pitch2 + j];
    }
  }
}

__global__ void matmul_kernel_ver3(
    const float * const mat1, const float * const mat2, float * const mat3,
    const size_t pitch1, const size_t pitch2, const size_t pitch3,
    const size_t size1, const size_t size2, const size_t size3) {
  const size_t block_i = blockIdx.y * blockDim.y;
  const size_t block_j = blockIdx.x * blockDim.x;
  const size_t i = block_i + threadIdx.y;
  const size_t j = block_j + threadIdx.x;
  __shared__ float localA[threadsPerBlock][threadsPerBlock];
  __shared__ float localB[threadsPerBlock][threadsPerBlock];
  if (size1 - block_i >= threadsPerBlock && size3 - block_j >= threadsPerBlock) {
    float tmp = 0.0;
    for (size_t k = 0; k < size2; k += threadsPerBlock) {
      if (size2 - k < threadsPerBlock) {
        for (size_t k2 = k; k2 < size2; ++k2) {
          tmp += mat1[i*pitch1 + k2] * mat2[k2*pitch2 + j];
        }
      } else {
        __syncthreads();
        localA[threadIdx.y][threadIdx.x] = mat1[i*pitch1 + (k + threadIdx.x)];
        localB[threadIdx.y][threadIdx.x] = mat2[(k + threadIdx.y)*pitch2 + j];
        __syncthreads();
        for (size_t k2 = 0; k2 < threadsPerBlock; ++k2) {
          tmp += localA[threadIdx.y][k2] * localB[k2][threadIdx.x];
        }
      }
    }
    mat3[i*pitch3 + j] = tmp;
  } else if (i < size1 && j < size3) {
    for (size_t k = 0; k < size2; ++k) {
      mat3[i*pitch3 + j] += mat1[i*pitch1 + k] * mat2[k*pitch2 + j];
    }
  }
}

constexpr unsigned int blockSize = 64;

#define FMA_IMPL(index, k2) \
{ \
  const float v = localA[threadIdx.y + threadsPerBlock * index][k2]; \
  tmp##index##0 += v * v10; \
  tmp##index##1 += v * v11; \
  tmp##index##2 += v * v12; \
  tmp##index##3 += v * v13; \
  tmp##index##4 += v * v14; \
  tmp##index##5 += v * v15; \
  tmp##index##6 += v * v16; \
  tmp##index##7 += v * v17; \
}

#define FMA_IMPL2(index, k2) \
{ \
  const float v = localA[threadIdx.y + threadsPerBlock * index][k2]; \
  if (j0 < size3) { tmp##index##0 += v * v10; } \
  if (j1 < size3) { tmp##index##1 += v * v11; } \
  if (j2 < size3) { tmp##index##2 += v * v12; } \
  if (j3 < size3) { tmp##index##3 += v * v13; } \
  if (j4 < size3) { tmp##index##4 += v * v14; } \
  if (j5 < size3) { tmp##index##5 += v * v15; } \
  if (j6 < size3) { tmp##index##6 += v * v16; } \
  if (j7 < size3) { tmp##index##7 += v * v17; } \
}

#define FMA(k2) \
{ \
  const float v10 = localB[k2][threadIdx.x]; \
  const float v11 = localB[k2][threadIdx.x + threadsPerBlock]; \
  const float v12 = localB[k2][threadIdx.x + threadsPerBlock*2]; \
  const float v13 = localB[k2][threadIdx.x + threadsPerBlock*3]; \
  const float v14 = localB[k2][threadIdx.x + threadsPerBlock*4]; \
  const float v15 = localB[k2][threadIdx.x + threadsPerBlock*5]; \
  const float v16 = localB[k2][threadIdx.x + threadsPerBlock*6]; \
  const float v17 = localB[k2][threadIdx.x + threadsPerBlock*7]; \
  FMA_IMPL(0, k2) \
  FMA_IMPL(1, k2) \
  FMA_IMPL(2, k2) \
  FMA_IMPL(3, k2) \
  FMA_IMPL(4, k2) \
  FMA_IMPL(5, k2) \
  FMA_IMPL(6, k2) \
  FMA_IMPL(7, k2) \
}

#define APPLY_IMPL(indexi, indexj) \
  if (j##indexj < size3) { \
    mat3[i##indexi * pitch3 + j##indexj] = tmp##indexi##indexj; \
  }

#define APPLY(index) \
  if (i##index < size1) { \
    APPLY_IMPL(index, 0) \
    APPLY_IMPL(index, 1) \
    APPLY_IMPL(index, 2) \
    APPLY_IMPL(index, 3) \
    APPLY_IMPL(index, 4) \
    APPLY_IMPL(index, 5) \
    APPLY_IMPL(index, 6) \
    APPLY_IMPL(index, 7) \
  }

__global__ void matmul_kernel_ver4(
    const float * const mat1, const float * const mat2, float * const mat3,
    const size_t pitch1, const size_t pitch2, const size_t pitch3,
    const size_t size1, const size_t size2, const size_t size3) {
  const int i0 = threadIdx.y + blockIdx.y * blockSize;
  const int j0 = threadIdx.x + blockIdx.x * blockSize;
  const int i1 = threadIdx.y + threadsPerBlock + blockIdx.y * blockSize;
  const int j1 = threadIdx.x + threadsPerBlock + blockIdx.x * blockSize;
  const int i2 = threadIdx.y + threadsPerBlock*2 + blockIdx.y * blockSize;
  const int j2 = threadIdx.x + threadsPerBlock*2 + blockIdx.x * blockSize;
  const int i3 = threadIdx.y + threadsPerBlock*3 + blockIdx.y * blockSize;
  const int j3 = threadIdx.x + threadsPerBlock*3 + blockIdx.x * blockSize;
  const int i4 = threadIdx.y + threadsPerBlock*4 + blockIdx.y * blockSize;
  const int j4 = threadIdx.x + threadsPerBlock*4 + blockIdx.x * blockSize;
  const int i5 = threadIdx.y + threadsPerBlock*5 + blockIdx.y * blockSize;
  const int j5 = threadIdx.x + threadsPerBlock*5 + blockIdx.x * blockSize;
  const int i6 = threadIdx.y + threadsPerBlock*6 + blockIdx.y * blockSize;
  const int j6 = threadIdx.x + threadsPerBlock*6 + blockIdx.x * blockSize;
  const int i7 = threadIdx.y + threadsPerBlock*7 + blockIdx.y * blockSize;
  const int j7 = threadIdx.x + threadsPerBlock*7 + blockIdx.x * blockSize;
  float tmp00 = 0.0, tmp01 = 0.0, tmp02 = 0.0, tmp03 = 0.0, tmp04 = 0.0, tmp05 = 0.0, tmp06 = 0.0, tmp07 = 0.0;
  float tmp10 = 0.0, tmp11 = 0.0, tmp12 = 0.0, tmp13 = 0.0, tmp14 = 0.0, tmp15 = 0.0, tmp16 = 0.0, tmp17 = 0.0;
  float tmp20 = 0.0, tmp21 = 0.0, tmp22 = 0.0, tmp23 = 0.0, tmp24 = 0.0, tmp25 = 0.0, tmp26 = 0.0, tmp27 = 0.0;
  float tmp30 = 0.0, tmp31 = 0.0, tmp32 = 0.0, tmp33 = 0.0, tmp34 = 0.0, tmp35 = 0.0, tmp36 = 0.0, tmp37 = 0.0;
  float tmp40 = 0.0, tmp41 = 0.0, tmp42 = 0.0, tmp43 = 0.0, tmp44 = 0.0, tmp45 = 0.0, tmp46 = 0.0, tmp47 = 0.0;
  float tmp50 = 0.0, tmp51 = 0.0, tmp52 = 0.0, tmp53 = 0.0, tmp54 = 0.0, tmp55 = 0.0, tmp56 = 0.0, tmp57 = 0.0;
  float tmp60 = 0.0, tmp61 = 0.0, tmp62 = 0.0, tmp63 = 0.0, tmp64 = 0.0, tmp65 = 0.0, tmp66 = 0.0, tmp67 = 0.0;
  float tmp70 = 0.0, tmp71 = 0.0, tmp72 = 0.0, tmp73 = 0.0, tmp74 = 0.0, tmp75 = 0.0, tmp76 = 0.0, tmp77 = 0.0;
  __shared__ float localA[blockSize][threadsPerBlock+1];
  __shared__ float localB[threadsPerBlock][blockSize];
  for (int k = 0; k < size2; k += threadsPerBlock) {
    if (size1 - blockIdx.y * blockSize < blockSize ||
        size2 - k < threadsPerBlock ||
        size3 - blockIdx.x * blockSize < blockSize) {
      const size_t ek = min((size_t)threadsPerBlock, size2 - k);
      __syncthreads();
      if (threadIdx.x < ek) {
        if (i0 < size1) {
          localA[threadIdx.y][threadIdx.x] = __ldg(mat1 + i0 * pitch1 + k + threadIdx.x);
        }
        if (i1 < size1) {
          localA[threadIdx.y + threadsPerBlock][threadIdx.x] = __ldg(mat1 + i1 * pitch1 + k + threadIdx.x);
        }
        if (i2 < size1) {
          localA[threadIdx.y + threadsPerBlock * 2][threadIdx.x] = __ldg(mat1 + i2 * pitch1 + k + threadIdx.x);
        }
        if (i3 < size1) {
          localA[threadIdx.y + threadsPerBlock * 3][threadIdx.x] = __ldg(mat1 + i3 * pitch1 + k + threadIdx.x);
        }
        if (i4 < size1) {
          localA[threadIdx.y + threadsPerBlock * 4][threadIdx.x] = __ldg(mat1 + i4 * pitch1 + k + threadIdx.x);
        }
        if (i5 < size1) {
          localA[threadIdx.y + threadsPerBlock * 5][threadIdx.x] = __ldg(mat1 + i5 * pitch1 + k + threadIdx.x);
        }
        if (i6 < size1) {
          localA[threadIdx.y + threadsPerBlock * 6][threadIdx.x] = __ldg(mat1 + i6 * pitch1 + k + threadIdx.x);
        }
        if (i7 < size1) {
          localA[threadIdx.y + threadsPerBlock * 7][threadIdx.x] = __ldg(mat1 + i7 * pitch1 + k + threadIdx.x);
        }
      }
      __syncthreads();
      for (size_t k2 = 0; k2 < ek; ++k2) {
        float v10 = 0.0, v11 = 0.0, v12 = 0.0, v13 = 0.0, v14 = 0.0, v15 = 0.0, v16 = 0.0, v17 = 0.0;
        if (j0 < size3) {
          v10 = __ldg(mat2 + (k+k2) * pitch2 + j0);
        }
        if (j1 < size3) {
          v11 = __ldg(mat2 + (k+k2) * pitch2 + j1);
        }
        if (j2 < size3) {
          v12 = __ldg(mat2 + (k+k2) * pitch2 + j2);
        }
        if (j3 < size3) {
          v13 = __ldg(mat2 + (k+k2) * pitch2 + j3);
        }
        if (j4 < size3) {
          v14 = __ldg(mat2 + (k+k2) * pitch2 + j4);
        }
        if (j5 < size3) {
          v15 = __ldg(mat2 + (k+k2) * pitch2 + j5);
        }
        if (j6 < size3) {
          v16 = __ldg(mat2 + (k+k2) * pitch2 + j6);
        }
        if (j7 < size3) {
          v17 = __ldg(mat2 + (k+k2) * pitch2 + j7);
        }
        if (i0 < size1) FMA_IMPL2(0, k2)
        if (i1 < size1) FMA_IMPL2(1, k2)
        if (i2 < size1) FMA_IMPL2(2, k2)
        if (i3 < size1) FMA_IMPL2(3, k2)
        if (i4 < size1) FMA_IMPL2(4, k2)
        if (i5 < size1) FMA_IMPL2(5, k2)
        if (i6 < size1) FMA_IMPL2(6, k2)
        if (i7 < size1) FMA_IMPL2(7, k2)
      }
    } else {
      __syncthreads();
      localB[threadIdx.y][threadIdx.x] = __ldg(mat2 + (k+threadIdx.y) * pitch2 + j0);
      localB[threadIdx.y][threadIdx.x + threadsPerBlock] = __ldg(mat2 + (k+threadIdx.y) * pitch2 + j1);
      localB[threadIdx.y][threadIdx.x + threadsPerBlock*2] = __ldg(mat2 + (k+threadIdx.y) * pitch2 + j2);
      localB[threadIdx.y][threadIdx.x + threadsPerBlock*3] = __ldg(mat2 + (k+threadIdx.y) * pitch2 + j3);
      localB[threadIdx.y][threadIdx.x + threadsPerBlock*4] = __ldg(mat2 + (k+threadIdx.y) * pitch2 + j4);
      localB[threadIdx.y][threadIdx.x + threadsPerBlock*5] = __ldg(mat2 + (k+threadIdx.y) * pitch2 + j5);
      localB[threadIdx.y][threadIdx.x + threadsPerBlock*6] = __ldg(mat2 + (k+threadIdx.y) * pitch2 + j6);
      localB[threadIdx.y][threadIdx.x + threadsPerBlock*7] = __ldg(mat2 + (k+threadIdx.y) * pitch2 + j7);
      localA[threadIdx.y][threadIdx.x] = __ldg(mat1 + i0 * pitch1 + k + threadIdx.x);
      localA[threadIdx.y + threadsPerBlock][threadIdx.x] = __ldg(mat1 + i1 * pitch1 + k + threadIdx.x);
      localA[threadIdx.y + threadsPerBlock * 2][threadIdx.x] = __ldg(mat1 + i2 * pitch1 + k + threadIdx.x);
      localA[threadIdx.y + threadsPerBlock * 3][threadIdx.x] = __ldg(mat1 + i3 * pitch1 + k + threadIdx.x);
      localA[threadIdx.y + threadsPerBlock * 4][threadIdx.x] = __ldg(mat1 + i4 * pitch1 + k + threadIdx.x);
      localA[threadIdx.y + threadsPerBlock * 5][threadIdx.x] = __ldg(mat1 + i5 * pitch1 + k + threadIdx.x);
      localA[threadIdx.y + threadsPerBlock * 6][threadIdx.x] = __ldg(mat1 + i6 * pitch1 + k + threadIdx.x);
      localA[threadIdx.y + threadsPerBlock * 7][threadIdx.x] = __ldg(mat1 + i7 * pitch1 + k + threadIdx.x);
      __syncthreads();
      FMA(0)
      FMA(1)
      FMA(2)
      FMA(3)
      FMA(4)
      FMA(5)
      FMA(6)
      FMA(7)
    }
  }
  APPLY(0)
  APPLY(1)
  APPLY(2)
  APPLY(3)
  APPLY(4)
  APPLY(5)
  APPLY(6)
  APPLY(7)
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
  size_t bv = (size1+threadsPerBlock-1) / threadsPerBlock;
  size_t bh = (size3+threadsPerBlock-1) / threadsPerBlock;
  dim3 block(threadsPerBlock, threadsPerBlock);
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
  size_t bv = (size1+threadsPerBlock-1) / threadsPerBlock;
  size_t bh = (size3+threadsPerBlock-1) / threadsPerBlock;
  dim3 block(threadsPerBlock, threadsPerBlock);
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

__host__ void matmul_gpu_ver4(const float* const mat1, const float * const mat2, float * const mat3,
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
  size_t bv = (size1+blockSize-1) / blockSize;
  size_t bh = (size3+blockSize-1) / blockSize;
  dim3 block(threadsPerBlock, threadsPerBlock);
  dim3 grid(bh, bv);
  handle_error(cudaEventRecord(start, 0));
  matmul_kernel_ver4<<<grid, block>>>(
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
