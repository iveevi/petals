#include "tensor.hpp"
#include "autograd.hpp"
#include "composition.hpp"
#include "ops.hpp"

// Static variables
decltype(Tensor::tagger) Tensor::tagger;

// Formatting tensors
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
	return header + string_data(t);
}
