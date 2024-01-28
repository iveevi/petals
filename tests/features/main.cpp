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
	// Tensor B = Tensor::randn({ 2, 4 });
	Tensor B = Tensor::ones({ 2, 4 });
	fmt::print("A: {}\nB: {}\n", A, B);

	fmt::print("Running optimization:\n");

	// DynamicDeferred dd = DynamicDeferred::from(&ops::sub, { A, B });
	// fmt::print("dd evaled: {}\n", dd);

	// TODO: Two methods for easier composition: chains (creates a new function)
	// and dynamic compute graphs (lazily evaluated, then backward on them())

	// Chain dnn = Chain::from({
	// 	new Linear(*Linear::from(2, 5)),
	// 	new ops::_relu(ops::relu),
	// 	new Linear(*Linear::from(5, 4)),
	// });

	// Chain dnn = Chain::from(Linear::from(2, 5), ops::relu, Linear::from(5, 4));
	Chain dnn = Linear::from(2, 5) >> ops::relu >> Linear::from(5, 4);

	// SGD opt = SGD::from(dnn.parameters());
	auto opt = Momentum::from(dnn.parameters());

	for (size_t i = 0; i < 2; i++) {
		Tensor out = dnn.forward(A);

		fmt::print("\nA: {}\n  > out: {}\n", A, out);
		auto delta = 2 * (out - B);
		static_assert(std::is_same_v <decltype(delta), DynamicDeferred>);
		// TODO: cache result with the same args?
		// TODO: print on dyanmic deferred
		fmt::print("  > delta: {}\n", delta);

		// TODO: parameters() function for every function
		// TODO: how to restrict gradients for only scalar outputs?
		Tape tape = Tape::from(dnn.parameters());

		{
			auto loss = sum(square(out - B));
			static_assert(std::is_same_v <decltype(loss), DynamicDeferred>);
			fmt::print("  > loss: {}\n", loss);

			// Tensor dnn_out = out;
			// Tensor sub_out = dnn_out - B;
			// // Tensor square_out = ops::square.forward(sub_out);
			// // Tensor sum_out = ops::sum.forward(square_out);
			//
			// Tensor square_out = square(sub_out);
			// Tensor sum_out = sum(square_out);
			//
			// fmt::print("  > loss: {}\n", sum_out);
			//
			// // TODO: test as a dynamic compute graph
			// // TODO: dynamic compute graph conversion only applies when using operator()/.defered_forward(); .forward() means immediate computation
			// Tensor delta_sum = Tensor::ones({});
			// Tensor delta_square = ops::sum.pullback_args({ square_out }, delta_sum, tape)[0];
			// Tensor delta_sub = ops::square.pullback_args({ sub_out }, delta_square, tape)[0];
			// Tensor delta_dnn = ops::sub.pullback_args({ dnn_out, B }, delta_sub, tape)[0];
			// Tensor delta_out = dnn.pullback(delta_dnn, tape)[0];
			//
			// opt.step(tape);
		}
	}

	// TODO: implement automatic gradient checking for all operators (test)
}
