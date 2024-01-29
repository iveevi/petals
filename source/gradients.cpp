#include "gradients.hpp"
#include "ops.hpp"

// Vanilla SGD optimizer
SGD SGD::from(const std::vector <Tensor *> &dst, float lr)
{
	std::unordered_map <long long int, Tensor *> destinations;
	for (Tensor *const t : dst)
		destinations[t->tag] = t;

	return SGD(destinations, lr);
}

void SGD::step(const Tape &tape)
{
	for (const auto &[tag, grad] : tape) {
		if (destinations.contains(tag)) {
			Tensor *t = destinations[tag];
			t->copy(*t - lr * grad);
		}
	}
}

// SGD with basic momentum
Momentum Momentum::from(const std::vector <Tensor *> &dst, float lr, float momentum)
{
	std::unordered_map <long long int, Tensor *> destinations;
	for (Tensor *const t : dst)
		destinations[t->tag] = t;

	Momentum m(destinations, lr);
	m.momentum = momentum;
	return m;
}

void Momentum::step(const Tape &tape)
{
	for (const auto &[tag, grad] : tape) {
		if (destinations.contains(tag)) {
			Tensor *t = destinations[tag];
			if (!velocity.contains(tag))
				velocity[tag] = Tensor::zeros_like(grad);

			const Tensor &m = velocity[tag];
			Tensor nm = momentum * m - lr * grad;
			t->copy(*t + nm);
		}
	}
}
