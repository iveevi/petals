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

void cpu_kernel_gemm(const Resource &, const Resource &, Resource &, size_t, size_t, size_t);
