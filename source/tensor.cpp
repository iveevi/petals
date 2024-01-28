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

// Operator overloads
DynamicDeferred operator*(float k, const Tensor &A)
{
	// TODO: memory management with functions
	// conditional_ptr <dellocate?>
	return DynamicDeferred::from(new ops::_scalek { ops::_scalek::from(k) }, { A });
}

DynamicDeferred operator+(float k, const Tensor &A)
{
	return DynamicDeferred::from(new ops::_addk { ops::_addk::from(k) }, { A });
}

DynamicDeferred operator+(const Tensor &A, const Tensor &B)
{
	return DynamicDeferred::from(&ops::add, { A, B });
}

DynamicDeferred operator-(const Tensor &A, const Tensor &B)
{
	return DynamicDeferred::from(&ops::sub, { A, B });
}

DynamicDeferred operator*(const Tensor &A, const Tensor &B)
{
	return DynamicDeferred::from(&ops::mul, { A, B });
}

DynamicDeferred operator/(const Tensor &A, const Tensor &B)
{
	return DynamicDeferred::from(&ops::div, { A, B });
}
