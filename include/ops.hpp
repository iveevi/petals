#include <limits>

#include "autograd.hpp"
#include "composition.hpp"
#include "tensor.hpp"

// Standard operations
namespace ops {

struct _add : Function {
	using Function::Function;

	Tensor forward_args(const tensor_list &ts) override {
		assert_nargs <2> (ts);
		const Tensor &A = ts[0];
		const Tensor &B = ts[1];

		// TODO: issue warning to logger
		if (A.shape != B.shape)
			return {};

		Tensor out = Tensor::blank_like(A);
		cpu_kernel_ewop <kadd> (A.buffer, B.buffer, out.buffer);
		return out;
	}
} static add("add");

struct _sub : Function {
	using Function::Function;

	Tensor forward_args(const tensor_list &ts) override {
		assert_nargs <2> (ts);
		const Tensor &A = ts[0];
		const Tensor &B = ts[1];

		// TODO: issue warning to logger
		if (A.shape != B.shape) {
			fmt::print("{} {} expected Tensors of equal shape, got {} and {} instead.\n",
					fmt::format(fmt::fg(fmt::rgb(0xFF8888)), "[petals]"),
					fmt::format(fmt::fg(fmt::rgb(0x8888FF)), "(_sub)"),
					*A.shape, *B.shape);
			return {};
		}

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
} static sub("sub");

struct _mul : Function {
	using Function::Function;

	Tensor forward_args(const tensor_list &ts) override {
		assert_nargs <2> (ts);
		const Tensor &A = ts[0];
		const Tensor &B = ts[1];

		// TODO: issue warning to logger
		if (A.shape != B.shape)
			return {};

		Tensor out = Tensor::blank_like(A);
		cpu_kernel_ewop <kmul> (A.buffer, B.buffer, out.buffer);
		return out;
	}
} static mul("mul");

struct _div : Function {
	using Function::Function;

	Tensor forward_args(const tensor_list &ts) override {
		assert_nargs <2> (ts);
		const Tensor &A = ts[0];
		const Tensor &B = ts[1];

		// TODO: issue warning to logger
		if (A.shape != B.shape)
			return {};

		Tensor out = Tensor::blank_like(A);
		cpu_kernel_ewop <kdiv> (A.buffer, B.buffer, out.buffer);
		return out;
	}
} static div("div");

struct _addk : Function {
	using Function::Function;

	double k;
	Tensor forward_args(const tensor_list &ts) override {
		assert_nargs <1> (ts);
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		for (size_t i = 0; i < A.buffer.elements; i++)
			out.buffer.ptr[i] = k + A.buffer.ptr[i];

		// TODO: transform kernel
		return out;
	}

	static _addk from(double k) {
		_addk s(fmt::format("add <{:.4f}>", k));
		s.k = k;
		return s;
	}
};

struct _scalek : Function {
	using Function::Function;

	double k;

	Tensor forward_args(const tensor_list &ts) override {
		assert_nargs <1> (ts);
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		for (size_t i = 0; i < A.buffer.elements; i++)
			out.buffer.ptr[i] = k * A.buffer.ptr[i];

		// TODO: transform kernel
		return out;
	}

	tensor_list pullback_args(const tensor_list &ts, const Tensor &delta, Tape &tape) const override {
		assert_nargs <1> (ts);
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		for (size_t i = 0; i < A.buffer.elements; i++)
			out.buffer.ptr[i] = k * delta.buffer.ptr[i];

		if (tape.contains(A.tag))
			tape[A.tag] = out;

		return { out };
	}

	static _scalek from(double k) {
		_scalek s(fmt::format("scale <{:.4f}>", k));
		s.k = k;
		return s;
	}
};

struct _square : Function {
	using Function::Function;

	Tensor forward_args(const tensor_list &ts) override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		cpu_kernel_ewop <kmul> (A.buffer, A.buffer, out.buffer);
		return out;
	}

	tensor_list pullback_args(const tensor_list &ts, const Tensor &delta, Tape &tape) const override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		for (size_t i = 0; i < A.buffer.elements; i++)
			out.buffer.ptr[i] = 2 * delta.buffer.ptr[i] * A.buffer.ptr[i];

		if (tape.contains(A.tag))
			tape[A.tag] = out;

		return { out };
	}
} static square("square");

struct _sqrt : Function {
	using Function::Function;

	Tensor forward_args(const tensor_list &ts) override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		// TODO: element wise unary kernel
		for (size_t i = 0; i < A.buffer.elements; i++) {
			double x = A.buffer.ptr[i];
			out.buffer.ptr[i] = std::sqrt(x);
		}

		return out;
	}
} static sqrt("sqrt");

struct _sum : Function {
	using Function::Function;

	size_t dim = 0;

	// TODO: dimension
	Tensor forward_args(const tensor_list &ts) override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank({});
		double sum = 0.0f;

		size_t elements = A.buffer.elements;
		for (size_t i = 0; i < elements; i++)
			sum += A.buffer.ptr[i];

		out.buffer.ptr[0] = sum;
		return out;
	}

	tensor_list pullback_args(const tensor_list &ts, const Tensor &delta, Tape &tape) const override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);

		size_t elements = A.buffer.elements;
		for (size_t i = 0; i < elements; i++)
			out.buffer.ptr[i] = delta.buffer.ptr[0];

		if (tape.contains(A.tag))
			tape[A.tag] = out;

		return { out };
	}
} static sum("sum");

// TODO: integer returning operations (e.g. argmax, argmin)

// Classic activations
struct _relu : Function {
	using Function::Function;

	Tensor forward_args(const tensor_list &ts) override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		for (size_t i = 0; i < A.shape->elements(); i++) {
			double x = A.buffer.ptr[i];
			out.buffer.ptr[i] = std::fmax(0, x);
		}

		return out;
	}

	tensor_list pullback_args(const tensor_list &ts, const Tensor &delta, Tape &tape) const override {
		const Tensor &A = ts[0];
		Tensor out = Tensor::blank_like(A);
		for (size_t i = 0; i < A.shape->elements(); i++) {
			double x = A.buffer.ptr[i];
			out.buffer.ptr[i] = delta.buffer.ptr[i] * ((x > 0) ? 1 : 0);
		}

		if (tape.contains(A.tag))
			tape[A.tag] = out;

		return { out };
	}
} static relu("relu");

struct _sigmoid : Function {
	using Function::Function;

	Tensor forward_args(const tensor_list &ts) override {
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
			double sigmoid = 1/(1 + std::exp(-A.buffer.ptr[i]));
			out.buffer.ptr[i] = delta.buffer.ptr[i] * sigmoid * (1 - sigmoid);
		}

		// fmt::print("delta out into softmax: {}\n", delta[0]);
		// fmt::print("delta in from softmax: {}\n", out[0]);

		if (tape.contains(A.tag))
			tape[A.tag] = out;

		return { out };
	}
} static sigmoid("sigmoid");

struct _softmax : Function {
	// TODO: mode for fused cross entropy or simplified derivative..
	using Function::Function;

	size_t dim = 0;

	// TODO: dimension
	Tensor forward_args(const tensor_list &ts) override {
		assert_nargs <1> (ts);
		const Tensor &A = ts[0];

		Tensor out = Tensor::blank_like(A);

		// fmt::print("input to softmax: {}\n", A[0]);

		size_t last_shape = A.shape.value()[-1];
		size_t outer_shape = A.shape->elements() / last_shape;
		for (size_t i = 0; i < outer_shape; i++) {
			// TODO: cache this line
			double max = -std::numeric_limits <double> ::max();
			for (size_t j = 0; j < last_shape; j++) {
				size_t index = i * last_shape + j;
				max = std::max(max, A.buffer.ptr[index]);
			}

			double sum = 0;
			for (size_t j = 0; j < last_shape; j++) {
				size_t index = i * last_shape + j;
				sum += std::exp(A.buffer.ptr[index] - max);
			}

			for (size_t j = 0; j < last_shape; j++) {
				size_t index = i * last_shape + j;
				out.buffer.ptr[index] = std::exp(A.buffer.ptr[index] - max) / sum;
			}
		}

		// fmt::print("softmax output: {}\n", out);

		return out;
	}

	// TODO: double check this...
	tensor_list pullback_args(const tensor_list &ts, const Tensor &delta, Tape &tape) const override {
		assert_nargs <1> (ts);
		const Tensor &A = ts[0];

		Tensor out = Tensor::blank_like(A);
		// fmt::print("input to softmax: {}\n", A[0]);
		// fmt::print("  > delta to softmax: {}\n", delta);

		size_t last_shape = A.shape.value()[-1];
		size_t outer_shape = A.shape->elements() / last_shape;
		for (size_t i = 0; i < outer_shape; i++) {
			double max = -std::numeric_limits <double> ::max();
			for (size_t j = 0; j < last_shape; j++) {
				size_t index = i * last_shape + j;
				max = std::max(max, A.buffer.ptr[index]);
			}

			double sum = 0;
			for (size_t j = 0; j < last_shape; j++) {
				size_t index = i * last_shape + j;
				sum += std::exp(A.buffer.ptr[index] - max);
			}

			// TODO: multiply by the detla...
			for (size_t j = 0; j < last_shape; j++) {
				size_t index = i * last_shape + j;
				double x = std::exp(A.buffer.ptr[index] - max);
				out.buffer.ptr[index] = delta.buffer.ptr[index] * x * (sum - x)/(sum * sum);

				// double s = x/sum;
				// out.buffer.ptr[index] = 0.0f;
				// for (size_t k = 0; k < last_shape; k++) {
				// 	size_t sub_index = i * last_shape + k;
				// 	float de = s * (1 - s);
				// 	if (k != j) {
				// 		float os = std::exp(A.buffer.ptr[sub_index] - max)/sum;
				// 		de = - s * os;
				// 	}
				//
				// 	out.buffer.ptr[index] += de;
				// }
				//
				// out.buffer.ptr[index] *= delta.buffer.ptr[index];
			}
		}

		// fmt::print("delta in from softmax: {}\n", out);

		if (tape.contains(A.tag))
			tape[A.tag] = out;

		return { out };
	}
} static softmax("softmax");

}

// TODO: ml namespace

// Machine learning utilities
struct Linear : Function {
	using Function::Function;

	size_t in;
	size_t out;
	bool bias;
	Tensor W; // Combines weight and bias into a single matrix

	std::vector <Tensor *> parameters() override {
		return { &W };
	}

	Tensor forward_args(const tensor_list &ts) override {
		const Tensor &A = ts[0];
		// fmt::print("input to linear: {}\n", A);
		Tensor X = A.reshape(-1, in);
		Shape pad_shape = *X.shape;
		pad_shape[-1] = 1;

		Tensor padding = Tensor::ones(pad_shape);
		Tensor gemm_in = Tensor::concat(X, padding, 1);

		Shape out_shape = *A.shape;
		out_shape[-1] = out;

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
			 X.buffer, Wt.buffer, gemm_int.buffer,
			 X.shape.value()[0],
			 X.shape.value()[1],
			 Wt.shape.value()[1]
		);

		auto slice = gemm_int.slice(0, int_shape[-1] - 1, int_shape.size() - 1);
		// fmt::print("delta out into linear: {}\n", delta[0]);
		// fmt::print("delta in from linear: {}\n", slice[0]);

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

			// TODO: weakly_optional shape
			cpu_kernel_gemm
			(
				 XA.buffer, XD.buffer, dW.buffer,
				 in + 1, XA.shape.value()[1], out
			);

			tape[W.tag] = dW;
		}

		return { slice };
	}

	// Construction
	static Linear from(size_t in, size_t out, bool bias = true) {
		// NOTE: The weight-bias matrix is in transposed form
		Linear dense(fmt::format("linear ({}x{}:{})", in, out, bias ? "bias" : "no bias"));
		dense.in = in;
		dense.out = out;
		dense.bias = bias;
		// dense.W = Tensor::randn({ in + bias, out });
		dense.W = Tensor::xavier(in + bias, out);
		return dense;
	}
};

// Soft requirements for lazy arguments
template <typename T>
concept autograd_friendly = std::is_same_v <Tensor, T>
		|| std::is_same_v <weakly_optional <Tensor>, T>
		|| std::is_same_v <DynamicDeferred, T>;


// Exporting these functions as lazy evaluations
template <typename ... Args>
DynamicDeferred sum(const Args & ...args) {
	std::initializer_list <std::variant <Tensor, DynamicDeferred>> ts { args... };
	return DynamicDeferred::from(nop_ptr(&ops::sum), ts);
}

template <typename ... Args>
DynamicDeferred square(const Args & ...args) {
	std::initializer_list <std::variant <Tensor, DynamicDeferred>> ts { args... };
	return DynamicDeferred::from(nop_ptr(&ops::square), ts);
}

template <typename ... Args>
DynamicDeferred sqrt(const Args & ...args) {
	std::initializer_list <std::variant <Tensor, DynamicDeferred>> ts { args... };
	return DynamicDeferred::from(nop_ptr(&ops::sqrt), ts);
}

// Activations
template <typename T>
requires autograd_friendly <T>
DynamicDeferred relu(const T &X) {
	return DynamicDeferred::from(nop_ptr(&ops::relu), { X });
}

template <typename T>
requires autograd_friendly <T>
DynamicDeferred sigmoid(const T &X) {
	return DynamicDeferred::from(nop_ptr(&ops::sigmoid), { X });
}

template <typename T>
requires autograd_friendly <T>
DynamicDeferred softmax(const T &X) {
	return DynamicDeferred::from(nop_ptr(&ops::softmax), { X });
}

// Operators
template <typename A>
requires autograd_friendly <A>
DynamicDeferred operator+(double k, const A &X)
{
	return DynamicDeferred::from(value_ptr(ops::_addk::from(k)), { X });
}

template <typename A>
requires autograd_friendly <A>
DynamicDeferred operator+(const A &X, double k)
{
	return DynamicDeferred::from(value_ptr(ops::_addk::from(k)), { X });
}

template <typename A>
requires autograd_friendly <A>
DynamicDeferred operator-(double k, const A &X)
{
	return DynamicDeferred::from(value_ptr(ops::_addk::from(-k)), { X });
}

template <typename A>
requires autograd_friendly <A>
DynamicDeferred operator-(const A &X, double k)
{
	return DynamicDeferred::from(value_ptr(ops::_addk::from(-k)), { X });
}

template <typename A>
requires autograd_friendly <A>
DynamicDeferred operator*(double k, const A &X)
{
	return DynamicDeferred::from(value_ptr(ops::_scalek::from(k)), { X });
}

template <typename A>
requires autograd_friendly <A>
DynamicDeferred operator/(const A &X, double k)
{
	return DynamicDeferred::from(value_ptr(ops::_scalek::from(1.0f/k)), { X });
}

// Binary operators
template <typename A, typename B>
requires autograd_friendly <A> && autograd_friendly <B>
DynamicDeferred operator+(const A &Xa, const B &Xb)
{
	return DynamicDeferred::from(nop_ptr(&ops::add), { Xa, Xb });
}

template <typename A, typename B>
requires autograd_friendly <A> && autograd_friendly <B>
DynamicDeferred operator-(const A &Xa, const B &Xb)
{
	return DynamicDeferred::from(nop_ptr(&ops::sub), { Xa, Xb });
}

template <typename A, typename B>
requires autograd_friendly <A> && autograd_friendly <B>
DynamicDeferred operator*(const A &Xa, const B &Xb)
{
	return DynamicDeferred::from(nop_ptr(&ops::mul), { Xa, Xb });
}

template <typename A, typename B>
requires autograd_friendly <A> && autograd_friendly <B>
DynamicDeferred operator/(const A &Xa, const B &Xb)
{
	return DynamicDeferred::from(nop_ptr(&ops::div), { Xa, Xb });
}
