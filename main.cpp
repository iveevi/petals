#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>

#include "resource.hpp"

// TODO: Lazy evaluation for both CPU and Vulkan mode

struct Shape : std::vector <long int> {
	using parent = std::vector <long int>;

	// Empty shape; a scalar
	Shape() {}

	// From a vector
	template <typename T>
	requires std::is_integral_v <T>
	Shape(const std::vector <T> &v) {
		resize(v.size());

		for (size_t i = 0; i < v.size(); i++)
			(*this)[i] = v[i];
	}

	// Construct from integral initializer list
	// TODO: use optional format, in case there are negative numbers
	template <std::integral T>
	Shape(const std::initializer_list <T> &list) {
		resize(list.size());

		size_t i = 0;
		for (auto it = list.begin(); it != list.end(); it++)
			(*this)[i++] = *it;
	}

	size_t elements() const {
		size_t prod = 1;
		for (size_t i = 0; i < size(); i++)
			prod *= (*this)[i];
		return prod;
	}

	Shape pop() const {
		if (size() < 1)
			return {};

		return std::vector <long int> (std::next(begin()), end());
	}

	// Indexing, with negatives allowed; returns the first index if out of bounds index
	const long int &operator[](int i) const {
		int j = (i > 0) ? i : i + size();
		// TODO: warning
		return (j < size() && j >= 0) ? parent::operator[](j) : parent::operator[](0);
	}

	long int &operator[](int i) {
		int j = (i > 0) ? i : i + size();
		// TODO: warning
		return (j < size() && j >= 0) ? parent::operator[](j) : parent::operator[](0);
	}

	// Reshape, with an optional ambiguous dimension (-1)
	std::optional <Shape> reshape(const Shape &other) const {
		Shape final = other;

		int minus = -1;
		size_t remainder = elements();

		size_t i = 0;
		while (i < other.size()) {
			if (other[i] == -1) {
				if (minus == -1) {
					minus = i++;
					continue;
				} else {
					// Cannot have multiple ambiguous dimensions
					return std::nullopt;
				}
			}

			if (remainder % other[i] != 0)
				return std::nullopt;

			remainder /= other[i++];
		}

		if (minus != -1)
			final[minus] = remainder;
		else if (remainder != 1)
			return std::nullopt;

		return final;
	}

	// Comparison
	bool operator==(const Shape &other) const {
		if (other.size() != size())
			return false;

		for (size_t i = 0; i < size(); i++) {
			if ((*this)[i] != other[i])
				return false;
		}

		return true;
	}
};

auto format_as(const Shape &s)
{
	std::string str = "(";
	for (size_t i = 0; i < s.size(); i++) {
		str += fmt::to_string(s[i]);
		if (i + 1 < s.size())
			str += ", ";
	}
	return str + ")";
}

// NOTE: To enable more semantic programming, we use this wrapper over the
// standard C++ optional type which can implicitly unpack into its underlying
// type. The value checking capabilities are still present, however.
template <typename T>
struct weakly_optional : std::optional <T> {
	using std::optional <T> ::optional;

	operator T() const {
		if (!this->has_value()) {
			// TODO: custom logging
			fmt::print("Implicitly converting weakly_optional tensor of null value\n");
			std::exit(EXIT_FAILURE);
		}

		return this->value();
	}

	// TODO: only if it has an indexing
	T::index_type operator[](int i) {
		// Implicitly unpack
		T value = *this;
		return value[i];
	}
};

struct Tensor {
	Resource buffer;
	std::optional <Shape> shape = std::nullopt; // TODO: make this weakly optional

	// Gradient tracking; only for backward pass
	// TODO: store gradients of all tensors in an operation separately...
	struct {
		bool enabled = false;
		std::optional <Resource> value;
	} grad;

	// Type definitions for templates
	using index_type = Tensor;

	// Assigning values
	Tensor &operator=(float v) {
		// TODO: type checking
		buffer.memset(v);
		return *this;
	}

	// Indexing the topmost dimension
	// TODO: negative dimensions as well...
	weakly_optional <Tensor> operator[](size_t i) const {
		if (!shape || i >= (*shape)[0])
			return std::nullopt;

		Shape sub_shape = shape->pop();
		size_t start = i * sub_shape.elements();
		size_t end = start + sub_shape.elements();
		Resource sub_buffer = buffer.slice(start, end).value();
		return Tensor { sub_buffer, sub_shape };
	}

	// Reshaping tensors
	weakly_optional <Tensor> reshape(const Shape &other) const {
		if (auto reshaped = shape->reshape(other)) {
			Resource reshaped_buffer = *buffer.slice(); // Gets the whole thing for free
			return Tensor { reshaped_buffer, *reshaped };
		}

		return std::nullopt;
	}

	template <std::integral ... Ts>
	weakly_optional <Tensor> reshape(Ts ... sizes) const {
		std::initializer_list <long int> other { (long int) sizes... };
		return reshape(other);
	}

	// Transposing 2D tensors
	weakly_optional <Tensor> transpose() const {
		if (shape->size() != 2)
			return std::nullopt;

		size_t rows = shape.value()[0];
		size_t cols = shape.value()[1];
		Shape transposed_shape { cols, rows };
		Tensor transposed = Tensor::blank(transposed_shape);

		// Write the elements
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++)
				transposed.buffer.ptr[j * rows + i] = buffer.ptr[i * cols + j];
		}

		return transposed;
	}

	// Cloning tensors; does not transfer tracking
	weakly_optional <Tensor> clone() const {
		Resource cloned_buffer = *buffer.clone();
		return Tensor { cloned_buffer, shape };
	}

	// Slicing through a single dimension
	weakly_optional <Tensor> slice(size_t start, size_t end, size_t dim = 0) {
		// TODO: allow negatives
		if (dim >= shape->size())
			return std::nullopt;

		// NOTE: End is not inclusive
		if (start >= end || start > shape.value()[0] || end > shape.value()[0])
			return std::nullopt;

		size_t prod_before = 1;
		size_t prod_after = 1;
		for (size_t i = 0; i < shape->size(); i++) {
			prod_before *= (i < dim) ? shape.value()[i] : 1;
			prod_after *= (i > dim) ? shape.value()[i] : 1;
		}

		Shape sliced_shape = *shape;
		sliced_shape[dim] = end - start;

		Tensor sliced = Tensor::blank(sliced_shape);
		for (size_t i = 0; i < prod_before; i++) {
			// TODO: put k in the inner loop?
			for (size_t j = 0; j < end - start; j++) {
				for (size_t k = 0; k < prod_after; k++) {
					size_t ithis = i * shape.value()[dim] * prod_after + j * prod_after + k;
					size_t isliced = i * (end - start) * prod_after + j * prod_after + k;
					sliced.buffer.ptr[isliced] = buffer.ptr[ithis];
				}
			}
		}

		return sliced;
	}

	// Enable gradient tracking; allocates the gradient value buffer ahead of time
	// TODO: maybe remove? set certain tensors to tracking by address with a gradient tracker type?
	weakly_optional <Tensor> tracked() const {
		Tensor tracked_tensor = *this;
		tracked_tensor.grad.enabled = true;
		if (auto grad_buffer = buffer.clone())
			tracked_tensor.grad.value = *grad_buffer;

		return std::nullopt;
	}

	// Blank tensor of a given shape; no memset-ing
	static weakly_optional <Tensor> blank(const Shape &shape, Resource::Type type = Resource::Type::f32, Resource::Device device = Resource::Device::eCPU) {
		if (auto buffer = Resource::from(shape.elements(), type, device))
			return Tensor { *buffer, shape };

		return std::nullopt;
	}

	// Zero tensor
	static weakly_optional <Tensor> zeros(const Shape &shape, Resource::Type type = Resource::Type::f32, Resource::Device device = Resource::Device::eCPU) {
		if (auto buffer = Resource::from(shape.elements(), type, device)) {
			buffer->memset(0.0f);
			return Tensor { *buffer, shape };
		}

		return std::nullopt;
	}

	static weakly_optional <Tensor> zeros_like(const Tensor &t) {
		if (auto buffer = Resource::from(t.shape.value().elements(), t.buffer.type, t.buffer.device)) {
			buffer->memset(0.0f);
			return Tensor { *buffer, t.shape.value() };
		}

		return std::nullopt;
	}

	// One tensor
	static weakly_optional <Tensor> ones(const Shape &shape, Resource::Type type = Resource::Type::f32, Resource::Device device = Resource::Device::eCPU) {
		if (auto buffer = Resource::from(shape.elements(), type, device)) {
			buffer->memset(1.0f);
			return Tensor { *buffer, shape };
		}

		return std::nullopt;
	}

	// Identity tensor
	// TODO: expand for multidim tensors
	static weakly_optional <Tensor> identity(size_t N, Resource::Type type = Resource::Type::f32, Resource::Device device = Resource::Device::eCPU) {
		Shape shape { N, N };
		if (auto buffer = Resource::from(shape.elements(), type, device)) {
			buffer->memset(0.0f);
			for (size_t i = 0; i < N; i++)
				buffer->ptr[i * N + i] = 1.0f;
			return Tensor { *buffer, shape };
		}

		return std::nullopt;
	}

	// Tensor concatenation; explicit dimension must be provided
	static weakly_optional <Tensor> concat(const Tensor &A, const Tensor &B, size_t dim) {
		// TODO: allow for negative dim (int)

		// Make sure the shapes match except for the provided dimension
		// TODO: unpack the shapes here
		if (A.shape->size() != B.shape->size())
			return std::nullopt;

		size_t sum = 0;
		size_t nA = 0;
		size_t nB = 0;
		size_t prod_before = 1;
		size_t prod_after = 1;
		for (size_t i = 0; i < A.shape->size(); i++) {
			if (i == dim) {
				nA = A.shape.value()[i];
				nB = B.shape.value()[i];
				sum = nA + nB;
			} else if (A.shape.value()[i] != B.shape.value()[i]) {
				return std::nullopt;
			} else {
				prod_before *= (sum == 0) ? A.shape.value()[i] : 1;
				prod_after *= (sum > 0) ? A.shape.value()[i] : 1;
			}
		}

		Shape cat_shape = *A.shape;
		cat_shape[dim] = sum;

		Tensor out = *Tensor::zeros(cat_shape);

		const Resource &rA = A.buffer;
		const Resource &rB = B.buffer;
		Resource &rout = out.buffer;

		for (size_t i = 0; i < prod_before; i++) {
			// TODO: put k in the inner loop?
			for (size_t j = 0; j < nA; j++) {
				for (size_t k = 0; k < prod_after; k++) {
					size_t iA = i * nA * prod_after + j * prod_after + k;
					size_t iout = i * sum * prod_after + j * prod_after + k;
					rout.ptr[iout] = rA.ptr[iA];
				}
			}

			for (size_t j = 0; j < nB; j++) {
				for (size_t k = 0; k < prod_after; k++) {
					size_t iB = i * nB * prod_after + j * prod_after + k;
					size_t iout = i * sum * prod_after + (j + nA) * prod_after + k;
					rout.ptr[iout] = rB.ptr[iB];
				}
			}
		}

		return out;
	}

	// TODO: stack (new dim) and cat (dim=-1)

	// Random tensor
	static weakly_optional <Tensor> randn(const Shape &shape, Resource::Type type = Resource::Type::f32, Resource::Device device = Resource::Device::eCPU) {
		if (auto buffer = Resource::from(shape.elements(), type, device)) {
			std::random_device rd;
			std::mt19937 generator(rd());
			std::uniform_real_distribution <> distribution(0, 1);
			for (size_t i = 0; i < shape.elements(); i++)
				buffer->ptr[i] = distribution(generator);
			return Tensor { *buffer, shape };
		}
		return std::nullopt;
	}

	static void drop(Tensor &t) {
		Resource::drop(t.buffer);
		t.shape = std::nullopt;
	}
};

std::string string_data(const Tensor &t)
{
	if (!t.shape)
		return "nil";

	if (t.shape.value().size() == 0) {
		// Single element
		float v = t.buffer.ptr[0];
		return fmt::format("{:.2f}", v);
	}

	std::string str = "[";
	// TODO: some method for shape.value()[X]
	for (size_t i = 0; i < t.shape.value()[0]; i++) {
		str += string_data(*t[i]);
		if (i + 1 < t.shape.value()[0])
			str += ", ";
	}

	return str + "]";
}

auto format_as(const Tensor &t)
{
	std::string header = "<Tensor: " + fmt::format("{}; {}; {}", *t.shape, t.buffer.type, t.buffer.device) + "> = ";
	return header + string_data(t);
}

// Autograd functions; note that function can only return a single tensor (tuples are expanded to separate entities)
using tensor_list = std::vector <Tensor>;
// using forward     = std::function <std::optional <Tensor> (const tensor_list &)>;
// using pushforward = std::function <Tensor (const tensor_list &)>;
// using pullback    = std::function <tensor_list (const Tensor &)>;

// TODO: this or struct with such functions
// struct Function {
// 	// TODO: implementations for each device...
//
// 	forward fwd;
// 	std::optional <pushforward> frule;
// 	std::optional <pullback> rrule;
//
// 	// TODO: some way to test each function
// };

struct Function {
	virtual ~Function() {}

	virtual weakly_optional <Tensor> forward_args(const tensor_list &) const = 0;

	// TODO: wrapper function to accept variadic list of tensors
	template <typename ... Args>
	// TODO: require tensorial
	weakly_optional <Tensor> forward(const Args & ...args) const {
		std::initializer_list <Tensor> ts { args... };
		return forward_args(ts);
	}
};

// Standard kernels
// TODO: namespace
enum ewop_mode {
	kadd,
	ksub
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
	}
}

// Standard functions
struct _add : Function {
	 weakly_optional <Tensor> forward_args(const tensor_list &ts) const override {
		// assert size = 2
		// TODO: check sizes here

		const Tensor &A = ts[0];
		const Tensor &B = ts[1];

		// TODO: issue warning to logger
		if (A.shape != B.shape)
			return std::nullopt;

		Tensor out = Tensor::zeros_like(A);
		cpu_kernel_ewop <kadd> (A.buffer, B.buffer, out.buffer);
		return out;
	}
} static const add;

struct _sub : Function {
	 weakly_optional <Tensor> forward_args(const tensor_list &ts) const override {
		// assert size = 2
		// TODO: check sizes here

		const Tensor &A = ts[0];
		const Tensor &B = ts[1];

		// TODO: issue warning to logger
		if (A.shape != B.shape)
			return std::nullopt;

		Tensor out = Tensor::zeros_like(A);
		cpu_kernel_ewop <ksub> (A.buffer, B.buffer, out.buffer);
		return out;
	}
} static const sub;

// Machine learning utilities

// TODO: chain(...) builds a composite Function

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

	weakly_optional <Tensor> forward_args(const tensor_list &ts) const override {
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

	weakly_optional <Tensor> pushforward(const tensor_list &ts) const {
		const Tensor &A = ts[0];
		if (auto X = A.reshape(-1, in)) {
			Shape pad_shape = *X->shape;
			pad_shape[-1] = 1;

			// TODO: only pad if there is a bias
			Tensor padding = Tensor::zeros(pad_shape);
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

		return std::nullopt;
	}

	// TODO: also need original inputs always
	// TODO: need a different structure for this...
	tensor_list pullback(const Tensor &delta) const {
		Tensor Wt = W.transpose();
		if (auto X = delta.reshape(-1, out)) {
			// TODO: different depending on the presence of the bias
			Shape int_shape = *delta.shape;
			int_shape[-1] = in + 1;

			Tensor gemm_int = Tensor::blank(int_shape);

			cpu_kernel_gemm
			(
				X->buffer, W.buffer, gemm_int.buffer,
				X->shape.value()[0],
				X->shape.value()[1],
				Wt.shape.value()[1]
			);

			return { gemm_int.slice(0, int_shape[-1] - 1, int_shape.size() - 1) };
		}

		return {};
	}

	// Construction
	static std::optional <Linear> from(size_t in, size_t out, bool bias = true) {
		// NOTE: The weight-bias matrix is in transposed form
		Linear dense;
		dense.in = in;
		dense.out = out;
		dense.bias = bias;
		dense.W = *Tensor::randn({ in + bias, out });
		return dense;
	}
};

// TODO: parameters() for computation graph returns a list of tensors with grad enabled only
// TODO: Computation graph, that is evaluated lazily... use auto normally
struct Node {
	const Function *const ftn;
	// Function ftn;
	// std::vector <Node> up;
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
	return add.forward(A, B);
}

weakly_optional <Tensor> operator-(const Tensor &A, const Tensor &B)
{
	return sub.forward(A, B);
}

int main()
{
	Tensor A = Tensor::randn({ 2, 2 });
	Tensor B = Tensor::randn({ 2, 4 });
	fmt::print("\nB: {}\n", B);

	fmt::print("Running optimization:\n");

	Linear dense = *Linear::from(2, 4);
	for (size_t i = 0; i < 100; i++) {
		Tensor out = dense.forward(A);
		fmt::print("\nA: {} -> out: {}\n", A, out);
		Tensor delta = 2.0f * (out - B);
		fmt::print("delta: {}\n", delta);
		Tensor delta_A = dense.pullback(delta)[0];
		A = A - 0.01f * delta_A;
	}
}
