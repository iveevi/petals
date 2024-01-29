#pragma once

#include <variant>

#include "autograd.hpp"

// Function composition via lazy evaluation
struct DynamicDeferred {
	Function *ftn;

	Tensor cached_eval;
	std::vector <Tensor> cached_args;
	std::vector <std::variant <Tensor, DynamicDeferred>> args;

	// TODO: optimize memory and compute by fusing operations and reusing buffers
	Tensor eval() {
		if (cached_args.size())
			return cached_eval;

		cached_args.clear();
		for (auto &v : args) {
			if (std::holds_alternative <Tensor> (v))
				cached_args.push_back(std::get <Tensor> (v));
			else
				cached_args.push_back(std::get <DynamicDeferred> (v));
		}

		cached_eval = ftn->forward_args(cached_args);
		return cached_eval;
	}

	operator Tensor() {
		return eval();
	}

	// TODO: backward/pullback
	// backward is pullback is delta of one

	// TODO: blacklisting certain deltas through the tape (e.g. target delta which is never used)
	tensor_list pullback(const Tensor &delta, Tape &tape) const {
		// TODO: warn here
		if (cached_args.empty()) {
			fmt::print("{} {} eval() must be performed in some manner before invoking pullback.\n",
					fmt::format(fmt::fg(fmt::rgb(0xFF8888)), "[petals]"),
					fmt::format(fmt::fg(fmt::rgb(0x8888FF)), "(dynamic deferred)"));
			return {};
		}

		tensor_list current_deltas = ftn->pullback_args(cached_args, delta, tape);
		tensor_list original_deltas;
		for (size_t i = 0; i < args.size(); i++) {
			const auto &v = args[i];
			if (std::holds_alternative <Tensor> (v)) {
				original_deltas.push_back(current_deltas[i]);
			} else {
				const DynamicDeferred &dd = std::get <DynamicDeferred> (v);
				tensor_list sub_deltas = dd.pullback(current_deltas[i], tape);
				original_deltas.insert(original_deltas.end(),
					sub_deltas.begin(), sub_deltas.end());
			}
		}

		return original_deltas;
	}

	tensor_list backward(Tape &tape) const {
		// TODO: check for dimension of output?
		return pullback(Tensor::ones({}), tape);
	}

	static DynamicDeferred from(Function *const ftn, const std::vector <std::variant <Tensor, DynamicDeferred>> &args) {
		DynamicDeferred dd;
		dd.ftn = ftn;
		dd.args = args;
		return dd;
	}

	static DynamicDeferred from_tensor_list(Function *const ftn, const std::vector <Tensor> &args) {
		std::vector <std::variant <Tensor, DynamicDeferred>> vargs;
		for (const auto &t : args)
			vargs.push_back(t);

		DynamicDeferred dd;
		dd.ftn = ftn;
		dd.args = vargs;
		return dd;
	}
};

// Function composition via chaining to construct a new function
struct Chain;

struct ChainProxy : std::vector <Function *> {
	using std::vector <Function *> ::vector;

	operator Chain();
};

// TODO: return a conditional ptr
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
	using Function::Function;

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

	// Evaluate as a lazy operation (recommended for typical ML)
	template <typename ... Args>
	DynamicDeferred operator()(const Args & ...args) {
		std::vector <std::variant <Tensor, DynamicDeferred>> ts { args... };
		return DynamicDeferred::from(this, ts);
	}

	tensor_list pullback_args(const tensor_list &args, const Tensor &delta, Tape &tape) const override {
		// Ensure that the arguments are the same
		// TODO: warn if not the case
		if (args.size() != node_args[0].size()) {
			fmt::print("{} {} arguments passed into pullback_args should be the same as those passed during forward.\n",
					fmt::format(fmt::fg(fmt::rgb(0xFF8888)), "[petals]"),
					fmt::format(fmt::fg(fmt::rgb(0x8888FF)), "(chain)"));
			return {};
		}

		for (size_t i = 0; i < args.size(); i++) {
			if (args[i].tag != node_args[0][i].tag) {
				// TODO: keep error codes of latest errors in the logger
				fmt::print("{} {} arguments passed into pullback_args should be the same as those passed during forward.\n",
						fmt::format(fmt::fg(fmt::rgb(0xFF8888)), "[petals]"),
						fmt::format(fmt::fg(fmt::rgb(0x8888FF)), "(chain)"));
				return {};
			}
		}

		// Do the pullback with cached inputs
		Tensor d = delta;
		for (long int i = nodes.size() - 1; i >= 0; i--)
			d = nodes[i]->pullback_args(node_args[i], d, tape)[0];

		return { d };
	}

	tensor_list pullback(const Tensor &delta, Tape &tape) const {
		return pullback_args(node_args[0], delta, tape);
	}

	// Generate string from list of functions
	static std::string to_string(const std::vector <Function *> &nodes) {
		std::string header = "chain [";
		for (size_t i = 0; i < nodes.size(); i++) {
			header += nodes[i]->tag;
			if (i + 1 < nodes.size())
				header += ", ";
		}
		return header + "]";
	}

	// Chain from already allocated functions
	static Chain from(const std::vector <Function *> &nodes) {
		Chain chain(to_string(nodes));
		chain.nodes = nodes;
		return chain;
	}

	// Chain by creating duplicate functions on the heap
	template <typename ... Args>
	static Chain from(const Args & ...args) {
		std::vector <Function *> nodes { auto_allocate(args)... };
		Chain chain(to_string(nodes));
		chain.nodes = nodes;
		return chain;
	}
};

// Printing utilities
std::string format_as(const DynamicDeferred &);
