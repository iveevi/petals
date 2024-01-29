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

bool buffer_close(const Resource &A, const Resource &B, float tolerance)
{
	if (A.elements != B.elements)
		return false;

	for (size_t i = 0; i < A.elements; i++) {
		if (std::abs(A.ptr[i] - B.ptr[i]) > tolerance) {
			fmt::print("delta is {} > {}\n", std::abs(A.ptr[i] - B.ptr[i]), tolerance);
			return false;
		}
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

// TODO: gradient checking with tape and lamda
// TODO: delta checking on arbitrary functions
bool test_linear()
{
	constexpr size_t WIDTH = 10;
	constexpr size_t HEIGHT = 15;
	constexpr size_t BATCH = 20;

	constexpr float epsilon = 1e-2f;

	Tensor X = Tensor::randn({ BATCH, WIDTH });
	Tensor Y = Tensor::randn({ BATCH, HEIGHT });
	Linear linear = Linear::from(WIDTH, HEIGHT);

	// Gradient checking on the parameters
	Tensor gt_dW = Tensor::zeros_like(linear.W);
	for (size_t i = 0; i < gt_dW.buffer.elements; i++) {
		// epsilon+
		Linear p_linear = linear;
		p_linear.W = linear.W.clone();
		p_linear.W.buffer.ptr[i] += epsilon;

		Tensor p_out = sum(square(p_linear.forward(X) - Y));

		// epsilon-
		Linear n_linear = linear;
		n_linear.W = linear.W.clone();
		n_linear.W.buffer.ptr[i] -= epsilon;

		Tensor n_out = sum(square(n_linear.forward(X) - Y));

		float dw = (p_out.buffer.ptr[0] - n_out.buffer.ptr[0])/(2 * epsilon);
		gt_dW.buffer.ptr[i] = dw;
	}

	fmt::print("FD dW = {}\n", gt_dW);

	Tape tape = Tape::from({ &linear.W });
	auto lY = linear.forward(X);
	auto loss = sum(square(lY - Y));
	loss.eval();

	auto deltas = loss.backward(tape);
	linear.pullback_args({ X }, deltas[0], tape);

	Tensor dW = tape[linear.W.tag];
	fmt::print("AD dW = {}\n", dW);

	return buffer_close(dW.buffer, gt_dW.buffer, epsilon * epsilon);
}

void robust_test(const std::function <bool ()> &test, size_t iterations = 100)
{
	for (size_t i = 0; i < iterations; i++) {
		if (!test()) {
			fmt::print("test failed\n");
			break;
		}
	}
}

int main()
{
	// robust_test(test_square);
	robust_test(test_linear);
	// test_linear();
}
