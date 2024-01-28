#pragma once

#include "tensor.hpp"
#include "gradients.hpp"
#include "kernels.hpp"

// Autograd functions; note that function can only return a single tensor (tuples are expanded to separate entities)
using tensor_list = std::vector <Tensor>;

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

	virtual tensor_list pullback_args(const tensor_list &, const Tensor &, Tape &) const {
		// TODO: give each function a name
		throw std::runtime_error("Function has not implemented pullback\n");
	}

	// Wrapper function to accept variadic list of tensors
	template <typename ... Args>
	weakly_optional <Tensor> forward(const Args & ...args) {
		std::initializer_list <Tensor> ts { args... };
		return forward_args(ts);
	}
};

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
