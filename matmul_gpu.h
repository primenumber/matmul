void matmul_gpu_ver1(const float* const mat1, const float* const mat2, float* const mat3,
    const size_t size1, const size_t size2, const size_t size3,
    const size_t pitch1, const size_t pitch2);

void matmul_gpu_ver2(const float* const mat1, const float* const mat2, float* const mat3,
    const size_t size1, const size_t size2, const size_t size3,
    const size_t pitch1, const size_t pitch2);

void matmul_gpu_ver3(const float* const mat1, const float* const mat2, float* const mat3,
    const size_t size1, const size_t size2, const size_t size3,
    const size_t pitch1, const size_t pitch2);
