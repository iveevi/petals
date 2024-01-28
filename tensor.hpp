#pragma once

#include <optional>
#include <random>
#include <vector>

#include <fmt/format.h>

#include "resource.hpp"

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
			// std::terminate();
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

struct Tensor {
	Resource buffer;
	std::optional <Shape> shape = std::nullopt; // TODO: make this weakly optional
	long long int tag = -1;

	// Tag generation
	static struct {
		long long int next_tag;

		long long int operator()() {
			return next_tag++;
		}
	} tagger;

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
		return Tensor { sub_buffer, sub_shape, tagger() };
	}

	// Reshaping tensors
	weakly_optional <Tensor> reshape(const Shape &other) const {
		if (auto reshaped = shape->reshape(other)) {
			Resource reshaped_buffer = *buffer.slice(); // Gets the whole thing for free
			return Tensor { reshaped_buffer, *reshaped, tagger() };
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
		return Tensor { cloned_buffer, shape, tagger() };
	}

	// Slicing through a single dimension
	weakly_optional <Tensor> slice(size_t start, size_t end, size_t dim = 0) {
		// TODO: allow negatives
		if (dim >= shape->size())
			return std::nullopt;

		// NOTE: End is not inclusive
		if (start >= end || start > shape.value()[dim] || end > shape.value()[dim])
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

	// Blank tensor of a given shape; no memset-ing
	static weakly_optional <Tensor> blank(const Shape &shape, Resource::Type type = Resource::Type::f32, Resource::Device device = Resource::Device::eCPU) {
		if (auto buffer = Resource::from(shape.elements(), type, device))
			return Tensor { *buffer, shape, tagger() };

		return std::nullopt;
	}

	static weakly_optional <Tensor> blank_like(const Tensor &t) {
		if (auto buffer = Resource::from(t.shape.value().elements(), t.buffer.type, t.buffer.device))
			return Tensor { *buffer, t.shape.value(), tagger() };

		return std::nullopt;
	}

	// Zero tensor
	static weakly_optional <Tensor> zeros(const Shape &shape, Resource::Type type = Resource::Type::f32, Resource::Device device = Resource::Device::eCPU) {
		if (auto buffer = Resource::from(shape.elements(), type, device)) {
			buffer->memset(0.0f);
			return Tensor { *buffer, shape, tagger() };
		}

		return std::nullopt;
	}

	static weakly_optional <Tensor> zeros_like(const Tensor &t) {
		if (auto buffer = Resource::from(t.shape.value().elements(), t.buffer.type, t.buffer.device)) {
			buffer->memset(0.0f);
			return Tensor { *buffer, t.shape.value(), tagger() };
		}

		return std::nullopt;
	}

	// One tensor
	static weakly_optional <Tensor> ones(const Shape &shape, Resource::Type type = Resource::Type::f32, Resource::Device device = Resource::Device::eCPU) {
		if (auto buffer = Resource::from(shape.elements(), type, device)) {
			buffer->memset(1.0f);
			return Tensor { *buffer, shape, tagger() };
		}

		return std::nullopt;
	}

	static weakly_optional <Tensor> ones_like(const Tensor &t) {
		if (auto buffer = Resource::from(t.shape.value().elements(), t.buffer.type, t.buffer.device)) {
			buffer->memset(1.0f);
			return Tensor { *buffer, t.shape.value(), tagger() };
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
			return Tensor { *buffer, shape, tagger() };
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
			return Tensor { *buffer, shape, tagger() };
		}
		return std::nullopt;
	}

	// TODO: manual or implicit/automatic?
	static void drop(Tensor &t) {
		Resource::drop(t.buffer);
		t.shape = std::nullopt;
	}
};

// Tagger for Tensors
decltype(Tensor::tagger) Tensor::tagger;

static std::string string_data(const Tensor &t)
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
