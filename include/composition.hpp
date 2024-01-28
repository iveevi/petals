#pragma once

#include <variant>

#include "autograd.hpp"

// Function composition via chaining to construct a new function
struct Chain;

struct ChainProxy : std::vector <Function *> {
	using std::vector <Function *> ::vector;

	operator Chain();
};

template <typename T>
requires std::is_base_of_v <Function, T>
static Function *auto_allocate(T t) {
	return new T(t);
}

template <typename A, typename B>
requires std::is_base_of_v <Function, A> && std::is_base_of_v <Function, B>
ChainProxy operator>>(const A &fa, const B &fb)
{
	return { auto_allocate(fa), auto_allocate(fb) };
}

template <typename T>
requires std::is_base_of_v <Function, T>
ChainProxy operator>>(const ChainProxy &cp, const T &ft)
{
	ChainProxy ret = cp;
	ret.push_back(auto_allocate(ft));
	return ret;
}

struct Chain : Function {
	// TODO: allow for non-linear chains
	std::vector <tensor_list> node_args;
	std::vector <Function *> nodes;

	// Get parameters from all nodes
	std::vector <Tensor *> parameters() override {
		std::vector <Tensor *> ps;
		for (Function *f : nodes) {
			const auto &fps = f->parameters();
			ps.insert(ps.end(), fps.begin(), fps.end());
		}

		return ps;
	}

	weakly_optional <Tensor> forward_args(const tensor_list &ts) override {
		node_args = { ts };

		Tensor out;
		for (size_t i = 0; i < nodes.size(); i++) {
			out = nodes[i]->forward_args(node_args.back());
			node_args.push_back({ out });
		}

		return out;
	}

	// NOTE: We cannot have the usual pullback here since interim
	// arguments are not given for the usual signature
	tensor_list pullback(const Tensor &delta, Tape &tape) const {
		Tensor d = delta;
		for (long int i = nodes.size() - 1; i >= 0; i--) {
			// TODO: careful when doing multi input pullbacks that matter (such as sub)
			d = nodes[i]->pullback_args(node_args[i], d, tape)[0];
		}

		return { d };
	}

	// TODO: override the message for pullback_args

	// Chain from already allocated functions
	static Chain from(const std::vector <Function *> &nodes) {
		Chain chain;
		chain.nodes = nodes;
		return chain;
	}

	// Chain by creating duplicate functions on the heap
	template <typename ... Args>
	static Chain from(const Args & ...args) {
		Chain chain;
		chain.nodes = { auto_allocate(args)... };
		return chain;
	}
};

// Function composition via lazy evaluation
struct DynamicDeferred {
	Function *ftn;
	std::vector <std::variant <Tensor, DynamicDeferred>> args;

	// TODO: backward/minimize/pullback

	// TODO: optimize memory and compute by fusing operations and reusing buffers
	operator Tensor() const {
		tensor_list evalled_args;
		for (const auto &v : args) {
			if (std::holds_alternative <Tensor> (v))
				evalled_args.push_back(std::get <Tensor> (v));
			else
				evalled_args.push_back(std::get <DynamicDeferred> (v));
		}

		return ftn->forward_args(evalled_args);
	}

	static DynamicDeferred from(Function *const ftn, const std::vector <std::variant <Tensor, DynamicDeferred>> &args) {
		return DynamicDeferred { ftn, args };
	}
};

// Operators
struct DynamicDeferred;

DynamicDeferred operator*(float, const Tensor &);
DynamicDeferred operator+(float, const Tensor &);

DynamicDeferred operator+(const Tensor &, const Tensor &);
DynamicDeferred operator-(const Tensor &, const Tensor &);
DynamicDeferred operator*(const Tensor &, const Tensor &);
DynamicDeferred operator/(const Tensor &, const Tensor &);
