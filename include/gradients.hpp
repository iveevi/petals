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
	float lr;

	Optimizer(const std::unordered_map <long long int, Tensor *> &dst, float alpha)
			: destinations(dst), lr(alpha) {}

	virtual void step(const Tape &) = 0;
};

struct SGD : Optimizer {
	using Optimizer::Optimizer;

	static SGD from(const std::vector <Tensor *> &, float = 0.01f);
	virtual void step(const Tape &) override;
};

struct Momentum : Optimizer {
	using Optimizer::Optimizer;

	std::unordered_map <long long int, Tensor> velocity;
	float momentum;

	static Momentum from(const std::vector <Tensor *> &, float = 0.01f, float = 0.9f);
	virtual void step(const Tape &) override;
};
