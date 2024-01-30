#pragma once

#include <unordered_map>

#include "tensor.hpp"

// Recording gradients during the pullback/backward
struct Tape : std::unordered_map <long long int, Tensor> {
	static Tape from(const std::vector <Tensor *> &tags) {
		Tape grads;
		for (Tensor *t: tags)
			grads[t->tag] = Tensor {};
		return grads;
	}
};

// Optimizers for applying gradients from the tape
struct Optimizer {
	std::unordered_map <long long int, Tensor *> destinations;
	double lr;

	Optimizer(const std::unordered_map <long long int, Tensor *> &dst, double alpha)
			: destinations(dst), lr(alpha) {}

	virtual void step(const Tape &) = 0;
};

struct SGD : Optimizer {
	using Optimizer::Optimizer;

	static SGD from(const std::vector <Tensor *> &, double = 0.01f);
	virtual void step(const Tape &) override;
};

struct Momentum : Optimizer {
	using Optimizer::Optimizer;

	std::unordered_map <long long int, Tensor> velocity;
	double momentum;

	static Momentum from(const std::vector <Tensor *> &, double = 0.01f, double = 0.9f);
	virtual void step(const Tape &) override;
};

struct Adam : Optimizer {
	using Optimizer::Optimizer;

	double beta1 = 0.0f;
	double beta2 = 0.0f;
	size_t iteration = 0;

	std::unordered_map <long long int, Tensor> M;
	std::unordered_map <long long int, Tensor> Mh;
	std::unordered_map <long long int, Tensor> S;
	std::unordered_map <long long int, Tensor> Sh;

	static Adam from(const std::vector <Tensor *> &, double = 0.01f, double = 0.9f, double = 0.999f);
	virtual void step(const Tape &) override;
};
