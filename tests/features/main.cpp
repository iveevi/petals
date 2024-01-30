#include <cstdlib>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>

#include "tensor.hpp"
#include "gradients.hpp"
#include "autograd.hpp"
#include "composition.hpp"
#include "ops.hpp"

// TODO: latex plotting and live displaying?

int main()
{
	// TODO: fixing the random seed
	Tensor A = Tensor::randn({ 2, 2 });
	Tensor B = Tensor::randn({ 2, 4 });
	fmt::print("A: {}\nB: {}\n", A, B);

	fmt::print("Running optimization:\n");

	// Chain dnn = Linear::from(2, 5) >> ops::relu >> Linear::from(5, 4);
	Chain dnn = Chain::from(Linear::from(2, 5), ops::relu, Linear::from(5, 4));

	auto opt = SGD::from(dnn.parameters());
	// auto opt = Momentum::from(dnn.parameters());
	// auto opt = Adam::from(dnn.parameters());

	for (size_t i = 0; i < 1; i++) {
		Tensor out = dnn.forward(A);

		fmt::print("\nA: {}\n  > out: {}\n", A, out);

		// TODO: how to restrict gradients for only scalar outputs?
		Tape tape = Tape::from(dnn.parameters());

		// auto loss = sum(square(dnn(A) - B));
		auto loss = sum(dnn(A));
		loss.eval();

		// fmt::print("  > loss graph:\n{}\n", loss);
		// fmt::print("  > loss: {}\n", loss.eval());
		loss.backward(tape);

		// TODO: warn if empty tensors
		opt.step(tape);
	}
}
