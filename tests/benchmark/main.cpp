#include <benchmark/benchmark.h>

#include "ops.hpp"

// Tensor generation
static void BM_randn(benchmark::State &state)
{
	for (auto _ : state)
		Tensor::randn({ 100, 100 });
}

BENCHMARK(BM_randn);

// Tensor unary operations
#define BM_Unary(ftn, ...)                                  \
	static void BM_##ftn(benchmark::State &state) {     \
		Tensor A = Tensor::randn({ __VA_ARGS__ });  \
		for (auto _ : state)                        \
			ops::ftn.forward(A);                \
	}                                                   \
	BENCHMARK(BM_##ftn);

BM_Unary(relu,    100, 100, 100)
BM_Unary(sigmoid, 100, 100, 100)
BM_Unary(softmax, 100, 100, 100)
BM_Unary(square,  100, 100, 100)
BM_Unary(sum,     100, 100, 100)

// Binary operations
#define BM_Binary(ftn, ...)                                 \
	static void BM_##ftn(benchmark::State &state) {     \
		Tensor A = Tensor::randn({ __VA_ARGS__ });  \
		Tensor B = Tensor::randn({ __VA_ARGS__ });  \
		for (auto _ : state)                        \
			ops::ftn.forward(A, B);             \
	}                                                   \
	BENCHMARK(BM_##ftn);

BM_Binary(add,    100, 100, 100)
BM_Binary(sub,    100, 100, 100)
BM_Binary(mul,    100, 100, 100)
BM_Binary(div,    100, 100, 100)

// Floating point combinations
#define BM_Unary_Custom(ftn, ...)                           \
	static void BM_##ftn(benchmark::State &state) {     \
		Tensor A = Tensor::randn({ __VA_ARGS__ });  \
		auto F = ops::ftn::from(1.0);               \
		for (auto _ : state)                        \
			F.forward(A);                       \
	}                                                   \
	BENCHMARK(BM_##ftn);

BM_Unary_Custom(_addk,   100, 100, 100);
BM_Unary_Custom(_scalek, 100, 100, 100);

// TODO: Pullbacks

// Machine learning
static void BM_linear(benchmark::State &state)
{
	// TODO: with and without bias
	Linear L = Linear::from(100, 100);
	Tensor A = Tensor::randn({ 100, 100 });
	for (auto _ : state)
		L.forward(A);
}

BENCHMARK(BM_linear);

BENCHMARK_MAIN();
