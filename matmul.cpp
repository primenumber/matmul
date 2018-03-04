#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <boost/timer/timer.hpp>
#include "matmul_common.hpp"
#include "matmul_gpu.h"
#include "matmul_cpu.hpp"

size_t ceil_power2(size_t x, size_t y) {
  return ((x - 1) | (y - 1)) + 1;
}

void matmul_naive(const float* const mat1, const float* const mat2, float* const mat3,
    const size_t size1, const size_t size2, const size_t size3,
    const size_t pitch1, const size_t pitch2) {
  for (size_t i = 0; i < size1; ++i) {
    for (size_t j = 0; j < size3; ++j) {
      mat3[i*pitch2 + j] = 0;
    }
  }
  for (size_t i = 0; i < size1; ++i) {
    for (size_t k = 0; k < size2; ++k) {
      for (size_t j = 0; j < size3; ++j) {
        mat3[i*pitch2 + j] += mat1[i*pitch1 + k] * mat2[k*pitch2 + j];
      }
    }
  }
}

float compare(const float* const mat1, const float* const mat2,
    const size_t row, const size_t col, const size_t pitch) {
  float diff = 0.0;
  for (size_t i = 0; i < row; ++i) {
    for (size_t j = 0; j < col; ++j) {
      diff += std::abs(mat1[i*pitch + j] - mat2[i*pitch + j]);
    }
  }
  return diff;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s SIZE1 SIZE2 SIZE3\n", argv[0]);
    exit(-1);
  }
  size_t size1 = std::atoi(argv[1]);
  size_t size2 = std::atoi(argv[2]);
  size_t size3 = std::atoi(argv[3]);
  bool verify = false;
  if (argc >= 5) {
    if (std::string(argv[4]) == "--verify") {
      verify = true;
    }
  }
  size_t alignment = 32;
  size_t ceil_y = alignment / sizeof(float);
  size_t pitch1 = ceil_power2(size2, ceil_y);
  size_t pitch2 = ceil_power2(size3, ceil_y);
  float* const mat1 = (float *)aligned_alloc(alignment, size1 * pitch1 * sizeof(float));
  float* const mat2 = (float *)aligned_alloc(alignment, size2 * pitch2 * sizeof(float));
  float* const mat3 = (float *)aligned_alloc(alignment, size1 * pitch2 * sizeof(float));
  //float* const mat3_g1 = (float *)aligned_alloc(alignment, size1 * pitch2 * sizeof(float));
  //float* const mat3_g2 = (float *)aligned_alloc(alignment, size1 * pitch2 * sizeof(float));
  float* const mat3_g3 = (float *)aligned_alloc(alignment, size1 * pitch2 * sizeof(float));
  float* const mat3_g4 = (float *)aligned_alloc(alignment, size1 * pitch2 * sizeof(float));
  initialize_matrix(mat1, size1, size2, pitch1);
  initialize_matrix(mat2, size2, size3, pitch2);
  boost::timer::cpu_timer timer;
  matmul_cpu(mat1, mat2, mat3, size1, size2, size3, pitch1, pitch2);
  boost::timer::cpu_times elapsed = timer.elapsed();
  std::fprintf(stderr, "Time: %fms, %f GFLOPS\n", elapsed.wall / 1e6, (float)size1*size2*size3*2 / elapsed.wall);
  std::fprintf(stderr, "cpu: time:%s", timer.format().c_str());
  //timer.start();
  //matmul_gpu_ver1(mat1, mat2, mat3_g1, size1, size2, size3, pitch1, pitch2);
  //std::fprintf(stderr, "gpu1: time:%s", timer.format().c_str());
  //timer.start();
  //matmul_gpu_ver2(mat1, mat2, mat3_g2, size1, size2, size3, pitch1, pitch2);
  //std::fprintf(stderr, "gpu2: time:%s", timer.format().c_str());
  timer.start();
  matmul_gpu_ver3(mat1, mat2, mat3_g3, size1, size2, size3, pitch1, pitch2);
  std::fprintf(stderr, "gpu3: time:%s", timer.format().c_str());
  timer.start();
  matmul_gpu_ver4(mat1, mat2, mat3_g4, size1, size2, size3, pitch1, pitch2);
  std::fprintf(stderr, "gpu4: time:%s", timer.format().c_str());
  float d;
  //d = compare(mat3_g1, mat3, size1, size3, pitch2);
  //std::fprintf(stderr, "diff cpu-gpu1: %f\n", d);
  //d = compare(mat3_g2, mat3, size1, size3, pitch2);
  //std::fprintf(stderr, "diff cpu-gpu2: %f\n", d);
  d = compare(mat3_g3, mat3, size1, size3, pitch2);
  std::fprintf(stderr, "diff cpu-gpu3: %f\n", d);
  d = compare(mat3_g4, mat3, size1, size3, pitch2);
  std::fprintf(stderr, "diff cpu-gpu4: %f\n", d);
  if (verify) {
    fprintf(stderr, "Verify start\n");
    float* const mat3_naive = (float *)aligned_alloc(alignment, size1 * pitch2 * sizeof(float));
    matmul_naive(mat1, mat2, mat3_naive, size1, size2, size3, pitch1, pitch2);
    d = compare(mat3, mat3_naive, size1, size3, pitch2);
    std::fprintf(stderr, "diff cpu: %f\n", d);
    d = compare(mat3_g4, mat3_naive, size1, size3, pitch2);
    std::fprintf(stderr, "diff gpu: %f\n", d);
  }
  free(mat1);
  free(mat2);
  free(mat3);
  //free(mat3_g1);
  //free(mat3_g2);
  free(mat3_g3);
  free(mat3_g4);
  return 0;
}
