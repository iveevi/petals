#pragma once

#include "resource.hpp"

// Standard kernels
enum ewop_mode {
	kadd,
	ksub,
	kmul,
	kdiv
};

template <ewop_mode op>
void cpu_kernel_ewop(const Resource &A, const Resource &B, Resource &C)
{
	// TODO: openmp
	for (size_t i = 0; i < A.elements; i++) {
		if constexpr (op == kadd)
			C.ptr[i] = A.ptr[i] + B.ptr[i];
		if constexpr (op == ksub)
			C.ptr[i] = A.ptr[i] - B.ptr[i];
		if constexpr (op == kmul)
			C.ptr[i] = A.ptr[i] * B.ptr[i];
		if constexpr (op == kdiv)
			C.ptr[i] = A.ptr[i] / B.ptr[i];
	}
}

void cpu_kernel_gemm(const Resource &A, const Resource &B, Resource &C, size_t N, size_t M, size_t K)
{
	// A is (N, M)
	// B is (M, K)
	// C is thus (N, K)

	// TODO: optimize
	for (size_t i = 0; i < N; i++) {
		for (size_t j = 0; j < K; j++) {
			float sum = 0.0f;
			for (size_t k = 0; k < M; k++) {
				float a = A.ptr[i * M + k];
				float b = B.ptr[k * K + j];
				sum += a * b;
			}

			C.ptr[i * K + j] = sum;
		}
	}
}
