#include "composition.hpp"

// Printing utilities
static std::string to_string(const DynamicDeferred &dd, size_t indents = 0)
{
	// TODO: show evaluation status
	std::string indentation = std::string(2 * indents, ' ');
	std::string header = fmt::format("{} {{\n", dd.ftn->tag);
	for (const auto &v : dd.args) {
		if (std::holds_alternative <Tensor> (v))
			header += fmt::format("{}  Tensor of shape {}\n", indentation, *std::get <Tensor> (v).shape);
		else
			header += fmt::format("{}  {}\n", indentation, to_string(std::get <DynamicDeferred> (v), indents + 1));
	}

	return header + indentation + "}";
}

std::string format_as(const DynamicDeferred &dd)
{
	// TODO: add name to each function
	return to_string(dd);
}

// Converting proxy chains
ChainProxy::operator Chain()
{
	std::vector <Function *> transfered(begin(), end());
	return Chain::from(transfered);
}
