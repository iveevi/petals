#include <cstdlib>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>

#include "resource.hpp"
#include "tensor.hpp"
#include "gradients.hpp"
#include "kernels.hpp"
#include "autograd.hpp"

// TODO: latex plotting and live displaying?

// Operator overloads
Tensor operator*(float k, const Tensor &A)
{
	// TODO: elements method for tensor
	Tensor out = A.clone();
	for (size_t i = 0; i < out.shape->elements(); i++)
		out.buffer.ptr[i] *= k;

	return out;
}

Tensor operator+(float k, const Tensor &A)
{
	// TODO: elements method for tensor
	Tensor out = A.clone();
	for (size_t i = 0; i < out.shape->elements(); i++)
		out.buffer.ptr[i] += k;

	return out;
}

weakly_optional <Tensor> operator+(const Tensor &A, const Tensor &B)
{
	return ops::add.forward(A, B);
}

weakly_optional <Tensor> operator-(const Tensor &A, const Tensor &B)
{
	return ops::sub.forward(A, B);
}

weakly_optional <Tensor> operator*(const Tensor &A, const Tensor &B)
{
	return ops::mul.forward(A, B);
}

// Function composition via chaining
// using ChainProxy = std::vector <Function *>;
struct Chain;

struct ChainProxy : std::vector <Function *> {
	using std::vector <Function *> ::vector;

	operator Chain();
};

template <typename T>
requires std::is_base_of_v <Function, T>
static Function *auto_allocate(T t) {
	return new T(t);
}

template <typename A, typename B>
requires std::is_base_of_v <Function, A> && std::is_base_of_v <Function, B>
ChainProxy operator>>(const A &fa, const B &fb)
{
	return { auto_allocate(fa), auto_allocate(fb) };
}

template <typename T>
requires std::is_base_of_v <Function, T>
ChainProxy operator>>(const ChainProxy &cp, const T &ft)
{
	ChainProxy ret = cp;
	ret.push_back(auto_allocate(ft));
	return ret;
}

struct Chain : Function {
	// TODO: allow for non-linear chains
	std::vector <tensor_list> node_args;
	std::vector <Function *> nodes;

	// Get parameters from all nodes
	std::vector <Tensor *> parameters() override {
		std::vector <Tensor *> ps;
		for (Function *f : nodes) {
			const auto &fps = f->parameters();
			ps.insert(ps.end(), fps.begin(), fps.end());
		}

		return ps;
	}

	weakly_optional <Tensor> forward_args(const tensor_list &ts) override {
		node_args = { ts };

		Tensor out;
		for (size_t i = 0; i < nodes.size(); i++) {
			out = nodes[i]->forward_args(node_args.back());
			node_args.push_back({ out });
		}

		return out;
	}

	// NOTE: We cannot have the usual pullback here since interim
	// arguments are not given for the usual signature
	tensor_list pullback(const Tensor &delta, Tape &tape) const {
		Tensor d = delta;
		for (long int i = nodes.size() - 1; i >= 0; i--) {
			// TODO: careful when doing multi input pullbacks that matter (such as sub)
			d = nodes[i]->pullback_args(node_args[i], d, tape)[0];
		}

		return { d };
	}

	// TODO: override the message for pullback_args

	// Chain from already allocated functions
	static Chain from(const std::vector <Function *> &nodes) {
		Chain chain;
		chain.nodes = nodes;
		return chain;
	}

	// Chain by creating duplicate functions on the heap
	template <typename ... Args>
	static Chain from(const Args & ...args) {
		Chain chain;
		chain.nodes = { auto_allocate(args)... };
		return chain;
	}

	// TODO: operator<< to chain easier
};

ChainProxy::operator Chain()
{
	std::vector <Function *> transfered(begin(), end());
	return Chain::from(transfered);
}

int main()
{
	// TODO: fixing the random seed
	Tensor A = Tensor::randn({ 2, 2 });
	// Tensor B = Tensor::randn({ 2, 4 });
	Tensor B = Tensor::ones({ 2, 4 });
	fmt::print("\nB: {}\n", B);

	fmt::print("Running optimization:\n");

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

	for (size_t i = 0; i < 10; i++) {
		Tensor out = dnn.forward(A);

		// Tensor out = dense1.forward(A);
		fmt::print("\nA: {}\n  > out: {}\n", A, out);
		Tensor delta = 2 * (out - B);

		// TODO: parameters() function for every function
		// TODO: how to restrict gradients for only scalar outputs?
		Tape tape = Tape::from(dnn.parameters());

		{
			Tensor dnn_out = out;
			Tensor sub_out = dnn_out - B;
			Tensor square_out = ops::square.forward(sub_out);
			Tensor sum_out = ops::sum.forward(square_out);
			fmt::print("  > loss: {}\n", sum_out);

			// TODO: test as a dynamic compute graph
			// TODO: dynamic compute graph conversion only applies when using operator()/.defered_forward(); .forward() means immediate computation
			Tensor delta_sum = Tensor::ones({});
			Tensor delta_square = ops::sum.pullback_args({ square_out }, delta_sum, tape)[0];
			Tensor delta_sub = ops::square.pullback_args({ sub_out }, delta_square, tape)[0];
			Tensor delta_dnn = ops::sub.pullback_args({ dnn_out, B }, delta_sub, tape)[0];
			Tensor delta_out = dnn.pullback(delta_dnn, tape)[0];

			opt.step(tape);
		}
	}

	// TODO: implement automatic gradient checking for all operators (test)
}
