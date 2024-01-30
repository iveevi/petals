#pragma once

#include <fmt/color.h>

#include "tensor.hpp"
#include "gradients.hpp"
#include "kernels.hpp"

// Autograd functions; note that function can only return a single tensor (tuples are expanded to separate entities)
using tensor_list = std::vector <Tensor>;

struct Function {
	std::string tag;

	Function(const std::string &str) : tag(str) {}
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
	virtual Tensor forward_args(const tensor_list &) = 0;

	virtual tensor_list pullback_args(const tensor_list &, const Tensor &, Tape &) const {
		// TODO: give each function a name
		throw std::runtime_error(fmt::format("Function ({}) has not implemented pullback\n", tag));
	}

	// Wrapper function to accept variadic list of tensors
	template <typename ... Args>
	Tensor forward(const Args & ...args) {
		std::initializer_list <Tensor> ts { args... };
		return forward_args(ts);
	}
};
