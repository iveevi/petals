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

	static SGD from(const std::vector <Tensor *> &dst, float lr = 0.01f) {
		std::unordered_map <long long int, Tensor *> destinations;
		for (Tensor *const t : dst)
			destinations[t->tag] = t;

		return SGD(destinations, lr);
	}

	virtual void step(const Tape &tape) override {
		for (const auto &[tag, grad] : tape) {
			if (destinations.contains(tag)) {
				Tensor *t = destinations[tag];
				t->copy(*t - lr * grad);
			}
		}
	}
};

struct Momentum : Optimizer {
	using Optimizer::Optimizer;

	std::unordered_map <long long int, Tensor> velocity;
	float momentum;

	static Momentum from(const std::vector <Tensor *> &dst, float lr = 0.01f, float momentum = 0.9f) {
		std::unordered_map <long long int, Tensor *> destinations;
		for (Tensor *const t : dst)
			destinations[t->tag] = t;

		Momentum m(destinations, lr);
		m.momentum = momentum;
		return m;
	}

	virtual void step(const Tape &tape) override {
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
};
