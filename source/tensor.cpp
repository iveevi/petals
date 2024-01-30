#include "tensor.hpp"
#include "autograd.hpp"
#include "composition.hpp"
#include "ops.hpp"

// Static variables
decltype(Tensor::tagger) Tensor::tagger;

// Formatting tensors
static std::string string_data(const double *const ptr, const std::optional <Shape> &opt_shape)
{
	if (!opt_shape)
		return "nil";

	Shape shape = opt_shape.value();
	if (shape.size() == 0) {
		// Single element
		double v = ptr[0];
		return fmt::format("{:.4f}", v);
	}

	Shape sub_shape = shape.pop();
	size_t sub_size = sub_shape.size();

	std::string str = "[";
	for (size_t i = 0; i < shape[0]; i++) {
		str += string_data(ptr + i * sub_size, sub_shape);
		if (i + 1 < shape[0])
			str += ", ";
	}

	return str + "]";
}

std::string format_as(const Shape &s)
{
	std::string str = "(";
	for (size_t i = 0; i < s.size(); i++) {
		str += fmt::to_string(s[i]);
		if (i + 1 < s.size())
			str += ", ";
	}
	return str + ")";
}

std::string format_as(const Tensor &t)
{
	std::string header = "<Tensor: " + fmt::format("{}; {}; {}", *t.shape, t.buffer.type, t.buffer.device) + "> = ";
	return header + string_data(t.buffer.ptr, t.shape);
}
