#include "tensor.hpp"
#include "autograd.hpp"

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
