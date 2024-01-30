#include "gradients.hpp"
#include "ops.hpp"

Optimizer::~Optimizer() {}

// Vanilla SGD optimizer
SGD SGD::from(const std::vector <Tensor *> &dst, double lr)
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
Momentum Momentum::from(const std::vector <Tensor *> &dst, double lr, double momentum)
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

			Tensor &m = velocity[tag];
			m = momentum * m - lr * grad;
			t->copy(*t + m);
		}
	}
}

// Adam
Adam Adam::from(const std::vector <Tensor *> &dst, double lr, double beta1, double beta2)
{
	std::unordered_map <long long int, Tensor *> destinations;
	for (Tensor *const t : dst)
		destinations[t->tag] = t;

	Adam m(destinations, lr);
	m.beta1 = beta1;
	m.beta2 = beta2;
	m.iteration = 0;
	return m;
}

void Adam::step(const Tape &tape)
{
	constexpr double epsilon = 1e-6f;

	iteration++;
	for (const auto &[tag, grad] : tape) {
		if (destinations.contains(tag)) {
			Tensor *t = destinations[tag];

			// Initialization if necessary
			if (!M.contains(tag))
				M[tag] = Tensor::zeros_like(grad);
			if (!Mh.contains(tag))
				Mh[tag] = Tensor::zeros_like(grad);
			if (!S.contains(tag))
				S[tag] = Tensor::zeros_like(grad);
			if (!Sh.contains(tag))
				Sh[tag] = Tensor::zeros_like(grad);

			// Update
			Tensor &Mx = M[tag];
			Tensor &Sx = S[tag];

			Mx = beta1 * Mx - (1 - beta1) * grad;
			Sx = beta2 * Sx + (1 - beta2) * (grad * grad);

			Tensor &Mhx = Mh[tag];
			Tensor &Shx = Sh[tag];

			Mhx = Mx / (1.0 - powf(beta1, iteration));
			Shx = Sx / (1.0 - powf(beta2, iteration));

			Tensor J = *t + (lr * Mhx) / sqrt(Shx + epsilon);

			t->copy(J);
		}
	}
}
