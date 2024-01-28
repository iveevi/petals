#include <cstdlib>
#include <optional>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>

#include "resource.hpp"
#include "tensor.hpp"

// TODO: Lazy evaluation for both CPU and Vulkan mode

// Autograd functions; note that function can only return a single tensor (tuples are expanded to separate entities)
using tensor_list = std::vector <Tensor>;
// using forward     = std::function <std::optional <Tensor> (const tensor_list &)>;
// using pushforward = std::function <Tensor (const tensor_list &)>;
// using pullback    = std::function <tensor_list (const Tensor &)>;

struct Tape : std::unordered_map <long long int, Tensor> {
	// std::unordered_map <long long int, Tensor> values;

	static Tape from(const std::vector <Tensor *> &tags) {
		Tape grads;
		for (Tensor *t: tags)
			grads[t->tag] = Tensor {};
		return grads;
	}
};

// TODO: printing each value...

struct Function {
	virtual ~Function() {}

	// Checking functions
	template <size_t N>
	[[gnu::always_inline]]
	static void assert_nargs(const tensor_list &args) {
		// TODO: log error then exit
		if (args.size() != N)
			throw std::runtime_error(fmt::format("Expected {} arguments, got {}\n", N, args.size()));
	}

	// Get the parameters of the function (as tags)
	virtual std::vector <Tensor *> parameters() {
		return {};
	}

	// Always need a way to get the primal value
	virtual weakly_optional <Tensor> forward_args(const tensor_list &) = 0;

	// TODO: change the output
	virtual tensor_list pullback_args(const tensor_list &, const Tensor &, Tape &) const {
		// TODO: give each function a name
		throw std::runtime_error("Function has not implemented pullback\n");
	}

	// TODO: wrapper function to accept variadic list of tensors
	template <typename ... Args>
	// TODO: require tensorial
	weakly_optional <Tensor> forward(const Args & ...args) {
		std::initializer_list <Tensor> ts { args... };
		return forward_args(ts);
	}
};

// Standard kernels
// TODO: namespace
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

// Standard operations
namespace ops {

struct _add : Function {
	 weakly_optional <Tensor> forward_args(const tensor_list &ts) override {
		assert_nargs <2> (ts);
		const Tensor &A = ts[0];
		const Tensor &B = ts[1];

		// TODO: issue warning to logger
		if (A.shape != B.shape)
			return std::nullopt;

		Tensor out = Tensor::blank_like(A);
		cpu_kernel_ewop <kadd> (A.buffer, B.buffer, out.buffer);
		return out;
	}
} static add;

struct _sub : Function {
	 weakly_optional <Tensor> forward_args(const tensor_list &ts) override {
		assert_nargs <2> (ts);
		const Tensor &A = ts[0];
		const Tensor &B = ts[1];

		// TODO: issue warning to logger
		if (A.shape != B.shape)
			return std::nullopt;

		Tensor out = Tensor::blank_like(A);
		cpu_kernel_ewop <ksub> (A.buffer, B.buffer, out.buffer);
		return out;
	}


	 tensor_list pullback_args(const tensor_list &ts, const Tensor &delta, Tape &tape) const override {
		assert_nargs <2> (ts);
		const Tensor &A = ts[0];
		const Tensor &B = ts[1];

		if (A.shape != B.shape)
			return {};

		Tensor outA = Tensor::blank_like(A);
		Tensor outB = Tensor::blank_like(B);

		for (size_t i = 0; i < A.shape->elements(); i++) {
			outA.buffer.ptr[i] = delta.buffer.ptr[i];
			outB.buffer.ptr[i] = -delta.buffer.ptr[i];
		}

		// Storing deltas
		// TODO: sum deltas or replace?
		if (tape.contains(A.tag))
			tape[A.tag] = outA;
		if (tape.contains(B.tag))
			tape[B.tag] = outB;

		return { outA, outB };
	 }
} static sub;

struct _mul : Function {
	 weakly_optional <Tensor> forward_args(const tensor_list &ts) override {
		assert_nargs <2> (ts);
		const Tensor &A = ts[0];
		const Tensor &B = ts[1];

		// TODO: issue warning to logger
		if (A.shape != B.shape)
			return std::nullopt;

		Tensor out = Tensor::blank_like(A);
		cpu_kernel_ewop <kmul> (A.buffer, B.buffer, out.buffer);
		return out;
	}
} static mul;

struct _div : Function {
	 weakly_optional <Tensor> forward_args(const tensor_list &ts) override {
		assert_nargs <2> (ts);
		const Tensor &A = ts[0];
		const Tensor &B = ts[1];

		// TODO: issue warning to logger
		if (A.shape != B.shape)
			return std::nullopt;

		Tensor out = Tensor::blank_like(A);
		cpu_kernel_ewop <kdiv> (A.buffer, B.buffer, out.buffer);
		return out;
	}
} static div;

struct _square : Function {
	 weakly_optional <Tensor> forward_args(const tensor_list &ts) override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		cpu_kernel_ewop <kmul> (A.buffer, A.buffer, out.buffer);
		return out;
	}

	 tensor_list pullback_args(const tensor_list &ts, const Tensor &delta, Tape &tape) const override {
		 // TODO: multiply as well...
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		for (size_t i = 0; i < A.shape->elements(); i++)
			out.buffer.ptr[i] = 2 * delta.buffer.ptr[i] * A.buffer.ptr[i];

		if (tape.contains(A.tag))
			tape[A.tag] = out;

		return { out };
	 }
} static square;

struct _sum : Function {
	size_t dim = 0;

	// TODO: dimension
	 weakly_optional <Tensor> forward_args(const tensor_list &ts) override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank({});
		float sum = 0.0f;
		for (size_t i = 0; i < A.shape->elements(); i++)
			sum += A.buffer.ptr[i];
		out.buffer.ptr[0] = sum;
		return out;
	}

	 tensor_list pullback_args(const tensor_list &ts, const Tensor &delta, Tape &tape) const override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		for (size_t i = 0; i < A.shape->elements(); i++)
			out.buffer.ptr[i] = delta.buffer.ptr[0];

		if (tape.contains(A.tag))
			tape[A.tag] = out;

		return { out };
	 }
} static sum;

// Classic activations
struct _relu : Function {
	 weakly_optional <Tensor> forward_args(const tensor_list &ts) override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		for (size_t i = 0; i < A.shape->elements(); i++) {
			float x = A.buffer.ptr[i];
			out.buffer.ptr[i] = std::fmax(0, x);
		}

		return out;
	}

	 tensor_list pullback_args(const tensor_list &ts, const Tensor &delta, Tape &tape) const override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		for (size_t i = 0; i < A.shape->elements(); i++) {
			float x = A.buffer.ptr[i];
			out.buffer.ptr[i] = delta.buffer.ptr[i] * ((x > 0) ? 1 : 0);
		}

		if (tape.contains(A.tag))
			tape[A.tag] = out;

		return { out };
	 }
} static relu;

struct _sigmoid : Function {
	 weakly_optional <Tensor> forward_args(const tensor_list &ts) override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		for (size_t i = 0; i < A.shape->elements(); i++)
			out.buffer.ptr[i] = 1/(1 + std::exp(-A.buffer.ptr[i]));

		return out;
	}

	 tensor_list pullback_args(const tensor_list &ts, const Tensor &delta, Tape &tape) const override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		for (size_t i = 0; i < A.shape->elements(); i++) {
			float sigmoid = 1/(1 + std::exp(-A.buffer.ptr[i]));
			out.buffer.ptr[i] = delta.buffer.ptr[i] * sigmoid * (1 - sigmoid);
		}

		if (tape.contains(A.tag))
			tape[A.tag] = out;

		return { out };
	 }
} static sigmoid;

}

// Machine learning utilities

// TODO: device
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

struct Linear : Function {
	size_t in;
	size_t out;
	bool bias;
	Tensor W; // Combines weight and bias into a single matrix

	std::vector <Tensor *> parameters() override {
		return { &W };
	}

	weakly_optional <Tensor> forward_args(const tensor_list &ts) override {
		const Tensor &A = ts[0];
		if (auto X = A.reshape(-1, in)) {
			Shape pad_shape = *X->shape;
			pad_shape[-1] = 1;

			Tensor padding = Tensor::ones(pad_shape);
			Tensor gemm_in = Tensor::concat(X, padding, 1);

			Shape out_shape = *A.shape;
			out_shape[-1] = out;

			// TODO: differentiate between zeros (memset) and blank (shape only, no memset)
			Tensor gemm_out = Tensor::blank(out_shape);

			cpu_kernel_gemm
			(
				gemm_in.buffer, W.buffer, gemm_out.buffer,
				gemm_in.shape.value()[0],
				gemm_in.shape.value()[1],
				W.shape.value()[1]
			);

			return gemm_out;
		}

		return std::nullopt;
	}

	// weakly_optional <Tensor> pushforward(const tensor_list &ts) const {
	// 	const Tensor &A = ts[0];
	// 	if (auto X = A.reshape(-1, in)) {
	// 		Shape pad_shape = *X->shape;
	// 		pad_shape[-1] = 1;
	//
	// 		// TODO: only pad if there is a bias
	// 		Tensor padding = Tensor::zeros(pad_shape);
	// 		Tensor gemm_in = Tensor::concat(X, padding, 1);
	//
	// 		Shape out_shape = *A.shape;
	// 		out_shape[-1] = out;
	//
	// 		Tensor gemm_out = Tensor::blank(out_shape);
	//
	// 		cpu_kernel_gemm
	// 		(
	// 			gemm_in.buffer, W.buffer, gemm_out.buffer,
	// 			gemm_in.shape.value()[0],
	// 			gemm_in.shape.value()[1],
	// 			W.shape.value()[1]
	// 		);
	//
	// 		return gemm_out;
	// 	}
	//
	// 	return std::nullopt;
	// }

	// TODO: also need original inputs always
	// TODO: need a different structure for this...
	tensor_list pullback_args(const tensor_list &ts, const Tensor &delta, Tape &tape) const override {
		const Tensor &A = ts[0];

		// TODO: skip the transpose
		Tensor Wt = W.transpose();
		Tensor X = delta.reshape(-1, out);
		// TODO: different depending on the presence of the bias
		Shape int_shape = *delta.shape;
		int_shape[-1] = in + 1;

		Tensor gemm_int = Tensor::blank(int_shape);

		cpu_kernel_gemm
		(
			X.buffer, W.buffer, gemm_int.buffer,
			X.shape.value()[0],
			X.shape.value()[1],
			Wt.shape.value()[1]
		);

		auto slice = gemm_int.slice(0, int_shape[-1] - 1, int_shape.size() - 1);

		// TODO: also check for W
		if (tape.contains(A.tag))
			tape[A.tag] = slice;

		if (tape.contains(W.tag)) {
			// TODO: outer product kernel
			Tensor XA = A.reshape(-1, in);
			Shape pad_shape = *XA.shape;
			pad_shape[-1] = 1;

			Tensor padding = Tensor::ones(pad_shape);
			XA = Tensor::concat(XA, padding, 1);
			XA = XA.transpose();

			Tensor XD = delta.reshape(-1, out);

			Tensor dW = Tensor::blank({ in + 1, out });

			// TODO: weakly_optional shape?
			cpu_kernel_gemm
			(
				XA.buffer, XD.buffer, dW.buffer,
				in + 1,
				XA.shape.value()[1],
				out
			);

			tape[W.tag] = dW;
		}

		return { slice };
	}

	// Construction
	static Linear from(size_t in, size_t out, bool bias = true) {
		// NOTE: The weight-bias matrix is in transposed form
		Linear dense;
		dense.in = in;
		dense.out = out;
		dense.bias = bias;
		dense.W = *Tensor::randn({ in + bias, out });
		return dense;
	}
};

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

	for (size_t i = 0; i < 100; i++) {
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

			Tensor delta_sum = Tensor::ones({});
			Tensor delta_square = ops::sum.pullback_args({ square_out }, delta_sum, tape)[0];
			Tensor delta_sub = ops::square.pullback_args({ sub_out }, delta_square, tape)[0];
			Tensor delta_dnn = ops::sub.pullback_args({ dnn_out, B }, delta_sub, tape)[0];
			Tensor delta_out = dnn.pullback(delta_dnn, tape)[0];

			fmt::print("Tape values:\n");
			for (const auto &[tag, value] : tape)
				fmt::print("  > {} -> {}\n", tag, value);

			// TODO: optimizer
			// NOTE: use += (inplace) so that a new tensor is not created
			for (Tensor *const t : dnn.parameters())
				*t = *t - 0.01f * tape[t->tag];

			// TODO: do minus eq
			// A = A - 0.01f * delta_out;
		}
	}

	// TODO: implement automatic gradient checking for all operators (test)
}
