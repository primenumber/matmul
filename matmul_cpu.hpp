#pragma once

void matmul_cpu(const float* const mat1, const float* const mat2, float* const mat3,
    const size_t size1, const size_t size2, const size_t size3,
    const size_t pitch1, const size_t pitch2);
