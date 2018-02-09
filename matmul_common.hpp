#pragma once
#include <algorithm>
#include <thread>
#include <vector>
#include <random>

template <typename T>
void initialize_matrix_worker(T* const ptr, size_t i, const size_t row, const size_t col, const size_t pitch, const size_t n) {
  std::random_device rd;
  std::minstd_rand rl(rd());
  std::normal_distribution<T> d;
  for (; i < row; i += n) {
    std::generate(ptr + i * pitch, ptr + i * pitch + col, [&]{ return d(rl); });
  }
}

template <typename T>
void initialize_matrix(T* const ptr, const size_t row, const size_t col, const size_t pitch) {
  std::vector<std::thread> vt;
  size_t n = std::thread::hardware_concurrency();
  for (size_t i = 0; i < n; ++i) {
    vt.emplace_back(initialize_matrix_worker<T>, ptr, i, row, col, pitch, n);
  }
  for (auto &&t : vt) t.join();
}
