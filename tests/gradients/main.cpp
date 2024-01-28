#include <functional>

#include <fmt/color.h>

#include "tensor.hpp"
#include "autograd.hpp"
#include "ops.hpp"

// TODO: put into a util
bool buffer_cheq(const Resource &A, const Resource &B)
{
	if (A.elements != B.elements)
		return false;

	for (size_t i = 0; i < A.elements; i++) {
		if (A.ptr[i] != B.ptr[i])
			return false;
	}

	return true;
}

bool test_square()
{
	Tape tape;
	Tensor X = Tensor::randn({ 3, 3 });
	Tensor S = ops::square.forward(X);
	Tensor dS = Tensor::ones_like(S);
	Tensor dX = ops::square.pullback_args({ X }, dS, tape)[0];

	// TODO: print only first time, except for the failed case
	// then run experiments again
	fmt::print("X: {}\nS: {}\ndS: {}\ndX: {}\n", X, S, dS, dX);

	Tensor gt_dX = 2 * X;
	fmt::print("gt dX: {}\n", gt_dX);

	// TODO: buffer check here
	return buffer_cheq(dX.buffer, gt_dX.buffer);
}

void robust_test(const std::function <bool ()> &test, size_t iterations = 100)
{
	for (size_t i = 0; i < iterations; i++) {
		if (!test())
			std::exit(EXIT_FAILURE);
	}
}

int main()
{
	robust_test(test_square);
	// test_square();
}
