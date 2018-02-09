#include <cmath>
#include <thread>
#include <mutex>
#include <future>
#include <vector>
#include <boost/timer/timer.hpp>
#include <immintrin.h>
#include "matmul_cpu.hpp"

constexpr size_t block_sizei = 128;
constexpr size_t block_sizej = 48;
constexpr size_t block_sizek = 32;
constexpr size_t block_sizej2 = 24;

void matmul_block(const float* const mat1, const float* const buf, float* const mat3,
    const size_t pitch1, const size_t pitch2) {
  for (size_t i = 0; i < block_sizei; i+=4) {
    float* const mat3i = mat3+i*pitch2;
    const float* const mat1i = mat1+i*pitch1;
    size_t index = 0;
    for (size_t j = 0; j < block_sizej; j+=block_sizej2) {
      __m256 c11 = _mm256_load_ps(mat3i         +j   );
      __m256 c12 = _mm256_load_ps(mat3i         +j+ 8);
      __m256 c13 = _mm256_load_ps(mat3i         +j+16);
      __m256 c21 = _mm256_load_ps(mat3i+  pitch2+j   );
      __m256 c22 = _mm256_load_ps(mat3i+  pitch2+j+ 8);
      __m256 c23 = _mm256_load_ps(mat3i+  pitch2+j+16);
      __m256 c31 = _mm256_load_ps(mat3i+2*pitch2+j   );
      __m256 c32 = _mm256_load_ps(mat3i+2*pitch2+j+ 8);
      __m256 c33 = _mm256_load_ps(mat3i+2*pitch2+j+16);
      __m256 c41 = _mm256_load_ps(mat3i+3*pitch2+j   );
      __m256 c42 = _mm256_load_ps(mat3i+3*pitch2+j+ 8);
      __m256 c43 = _mm256_load_ps(mat3i+3*pitch2+j+16);
      for (size_t k = 0; k < block_sizek; k+=1) {
        __m256 b1 = _mm256_load_ps(buf+index);
        __m256 b2 = _mm256_load_ps(buf+index+8);
        __m256 b3 = _mm256_load_ps(buf+index+16);
        index += block_sizej2;

        const float* const mat1ik = mat1i+k;
        __m256 a = _mm256_broadcast_ss(mat1ik);
        c11 = _mm256_fmadd_ps(a, b1, c11);
        c12 = _mm256_fmadd_ps(a, b2, c12);
        c13 = _mm256_fmadd_ps(a, b3, c13);
        a = _mm256_broadcast_ss(mat1ik+  pitch1);
        c21 = _mm256_fmadd_ps(a, b1, c21);
        c22 = _mm256_fmadd_ps(a, b2, c22);
        c23 = _mm256_fmadd_ps(a, b3, c23);
        a = _mm256_broadcast_ss(mat1ik+2*pitch1);
        c31 = _mm256_fmadd_ps(a, b1, c31);
        c32 = _mm256_fmadd_ps(a, b2, c32);
        c33 = _mm256_fmadd_ps(a, b3, c33);
        a = _mm256_broadcast_ss(mat1ik+3*pitch1);
        c41 = _mm256_fmadd_ps(a, b1, c41);
        c42 = _mm256_fmadd_ps(a, b2, c42);
        c43 = _mm256_fmadd_ps(a, b3, c43);
      }
      _mm256_store_ps(mat3i         +j   , c11);
      _mm256_store_ps(mat3i         +j+ 8, c12);
      _mm256_store_ps(mat3i         +j+16, c13);
      _mm256_store_ps(mat3i+  pitch2+j   , c21);
      _mm256_store_ps(mat3i+  pitch2+j+ 8, c22);
      _mm256_store_ps(mat3i+  pitch2+j+16, c23);
      _mm256_store_ps(mat3i+2*pitch2+j   , c31);
      _mm256_store_ps(mat3i+2*pitch2+j+ 8, c32);
      _mm256_store_ps(mat3i+2*pitch2+j+16, c33);
      _mm256_store_ps(mat3i+3*pitch2+j   , c41);
      _mm256_store_ps(mat3i+3*pitch2+j+ 8, c42);
      _mm256_store_ps(mat3i+3*pitch2+j+16, c43);
    }
  }
}

void matmul_worker(const float* const mat1, const float* const mat2, float* const mat3,
    const size_t size1, const size_t size2, const size_t size3,
    const size_t pitch1, const size_t pitch2, float * const buf) {
  for (size_t j = 0; j < size3; j += block_sizej) {
    for (size_t k = 0; k < size2; k += block_sizek) {
      if (size1 < block_sizei || size3 - j < block_sizej || size2 - k < block_sizek) {
        size_t ei = std::min(block_sizei, size1);
        size_t ej = std::min(block_sizej, size3 - j);
        size_t ek = std::min(block_sizek, size2 - k);
        for (size_t i2 = 0; i2 < ei; ++i2) {
          for (size_t k2 = 0; k2 < ek; ++k2) {
            for (size_t j2 = 0; j2 < ej; ++j2) {
              mat3[i2*pitch2 + j+j2] += mat1[i2*pitch1 + (k+k2)] * mat2[(k+k2)*pitch2 + j+j2];
            }
          }
        }
      } else {
        size_t index = 0;
        for (size_t j2 = 0; j2 < block_sizej; j2+=block_sizej2) {
          for (size_t k2 = 0; k2 < block_sizek; ++k2) {
            __m256 vec1 = _mm256_load_ps(mat2 + (k+k2)*pitch2 + j+j2);
            __m256 vec2 = _mm256_load_ps(mat2 + (k+k2)*pitch2 + j+j2+8);
            __m256 vec3 = _mm256_load_ps(mat2 + (k+k2)*pitch2 + j+j2+16);
            _mm256_store_ps(buf + index     , vec1);
            _mm256_store_ps(buf + index +  8, vec2);
            _mm256_store_ps(buf + index + 16, vec3);
            index += block_sizej2;
          }
        }
        matmul_block(
            mat1 + k,
            buf,
            mat3 + j,
            pitch1, pitch2);
      }
    }
  }
}

void matmul_cpu(const float* const mat1, const float* const mat2, float* const mat3,
    const size_t size1, const size_t size2, const size_t size3,
    const size_t pitch1, const size_t pitch2) {
  for (size_t i = 0; i < size1; ++i) {
    for (size_t j = 0; j < size3; ++j) {
      mat3[i*pitch2 + j] = 0;
    }
  }
  std::vector<std::thread> vt;
  std::atomic_size_t i(0);
  for (size_t t = 0; t < std::thread::hardware_concurrency()*2; ++t) {
    vt.emplace_back([&]() {
        float * const buf = (float*)aligned_alloc(32, sizeof(float) * block_sizej * block_sizek);
        while(true) {
          size_t myi = i += block_sizei;
          myi -= block_sizei;
          if (myi >= size1) break;
          matmul_worker(mat1 + myi * pitch1, mat2, mat3 + myi * pitch2, std::min(block_sizei, size1 - myi), size2, size3, pitch1, pitch2, buf);
        }
    });
  }
  for (auto &&t : vt) t.join();
}
